import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { analyze } from '../src/search.js';
import { createPosition, formatAction, generateActions, inCheck, positionKey, validateAction } from '../src/rules.js';

const defaultSuite = new URL('../examples/matches/suite.json', import.meta.url);
const defaults = { maxNodes: 10000, maxPlies: 40, maxDepth: 3, quiescenceDepth: 1, timeMs: 30000, terminalWork: 20000 };

/** Exhaust all legal submitted turns, without the search's pruning or policy. */
export function certifyTerminal(position, { terminalWork = defaults.terminalWork, timeMs = defaults.timeMs, terminalTimeMs = timeMs } = {}) {
  let work = 0;
  const exhausted = new Error('Terminal verification limit.');
  const deadline = performance.now() + terminalTimeMs;
  let reason = 'terminal-work-limit';
  const iterator = generateActions(position, {
    pruneUnsafe: false, cacheMoves: false, skipOptionalSpatial: false, firstOnly: true,
    tick() {
      if (++work > terminalWork) throw exhausted;
      if (performance.now() >= deadline) { reason = 'terminal-time-limit'; throw exhausted; }
    },
  });
  try {
    if (!iterator.next().done) return { terminal: false, verified: true, work };
    const checked = inCheck(position);
    return { terminal: true, verified: true, status: checked ? 'checkmate' : 'stalemate',
      winnerColor: checked ? 1 - position.action % 2 : null, inCheck: checked, work };
  } catch (error) {
    if (error !== exhausted) throw error;
    return { terminal: false, verified: false, work, reason };
  } finally { iterator.return(); }
}

function limitsFor(options) {
  const limits = { ...defaults, ...options };
  limits.terminalTimeMs = options.terminalTimeMs ?? limits.timeMs;
  // Zero is the transformer's dynamic horizon. Classical match callers keep
  // the normal positive-depth contract unless they explicitly select that engine.
  const minimumDepth = options.engine === 'transformer' ? 0 : 1;
  for (const [name, min, max] of [
    ['maxNodes', 0, 1e9], ['maxPlies', 0, 10000], ['maxDepth', minimumDepth, 64],
    ['quiescenceDepth', 0, 8], ['timeMs', 1, 3600000], ['terminalTimeMs', 1, 3600000], ['terminalWork', 0, 1e9],
  ]) if (!Number.isInteger(limits[name]) || limits[name] < min || limits[name] > max) throw new Error(`Invalid ${name}.`);
  return { ...Object.fromEntries(Object.keys(defaults).map(key => [key, limits[key]])), terminalTimeMs: limits.terminalTimeMs,
    ...(options.engine === 'transformer' ? { engine: 'transformer' } : {}) };
}

function recordSearch(result) {
  return Object.fromEntries([
    'bestAction', 'pv', 'status', 'completed', 'stoppedReason', 'score', 'scoreType', 'mateIn',
    'engine', 'depth', 'searchingDepth', 'rootActionsSearched', 'selectiveDepth',
    'effectiveQuiescenceDepth', 'nodes', 'searchNodes', 'generationNodes', 'qnodes',
    'ttHits', 'qTtHits', 'cutoffs', 'searchPolicy', 'policyLeaves', 'elapsedMs', 'limits',
    'candidateLimit', 'innerCandidateLimit', 'candidateCaps', 'candidateCacheEntries',
    'evaluations', 'inferenceBatches', 'mateProven', 'terminalProof',
  ].filter(key => result[key] !== undefined).map(key => [key, structuredClone(result[key])]));
}

/** Missing moves are unfinished games; legal budget fallbacks can still play. */
export async function runGame({ position, engineA = analyze, engineB = analyze, aColor = 0, playOnTimeLimit = false, ...options }) {
  const limits = limitsFor(options);
  if (![0, 1].includes(aColor)) throw new Error('aColor must be 0 (White) or 1 (Black).');
  if (typeof playOnTimeLimit !== 'boolean') throw new Error('playOnTimeLimit must be a boolean.');
  let current = structuredClone(position);
  const game = { aColor, white: aColor === 0 ? 'A' : 'B', black: aColor === 1 ? 'A' : 'B',
    result: 'UNFINISHED', winner: null, winnerColor: null, valid: true,
    reason: null, initialPosition: structuredClone(position), initialKey: positionKey(position), limits,
    playOnTimeLimit, moves: [] };
  const finish = (reason, extra = {}) => Object.assign(game, { reason, plies: game.moves.length,
    finalPosition: structuredClone(current), finalKey: positionKey(current), ...extra });
  const finishTerminal = (certificate, extra = {}) => {
    const winner = certificate.winnerColor === null ? null : certificate.winnerColor === aColor ? 'A' : 'B';
    return finish(certificate.status, { result: winner ? `${winner}_WIN` : 'DRAW', winner,
      winnerColor: certificate.winnerColor, certificate, ...extra });
  };
  for (let ply = 0; ply <= limits.maxPlies; ply++) {
    if (ply === limits.maxPlies) {
      let certificate;
      try { certificate = certifyTerminal(current, limits); }
      catch (error) { return finish('terminal-verification-error', { valid: false, error: error.message }); }
      if (!certificate.verified) return finish(certificate.reason, { certificate });
      if (certificate.terminal) return finishTerminal(certificate);
      return finish('ply-limit');
    }
    const color = current.action % 2, engine = color === aColor ? 'A' : 'B';
    // Give the engine its own copy and check the entire object, including promotions.
    const input = structuredClone(current), before = structuredClone(input);
    let result;
    try { result = await (engine === 'A' ? engineA : engineB)(input, { ...limits }); }
    catch (error) { return finish('engine-error', { valid: false, error: error.message, stoppedEngine: engine }); }
    if (!isDeepStrictEqual(input, before)) return finish('input-mutation', { valid: false, stoppedEngine: engine });
    if (!result || typeof result !== 'object') return finish('missing-result', { valid: false, stoppedEngine: engine });
    let search;
    try { search = recordSearch(result); }
    catch (error) { return finish('invalid-result', { valid: false, stoppedEngine: engine, error: error.message }); }
    const stop = (reason, extra = {}) => finish(reason, { stoppedEngine: engine, lastSearch: search, ...extra });
    if ([result.nodes, result.searchNodes, result.generationNodes].some(value => !Number.isInteger(value) || value < 0)
      || result.nodes > limits.maxNodes
      || result.nodes !== result.searchNodes + result.generationNodes) return stop('invalid-work-accounting', { valid: false });
    const terminalClaim = ['checkmate', 'stalemate'].includes(result.status);
    let next;
    if (result.bestAction !== null && result.bestAction !== undefined) {
      try { next = validateAction(current, result.bestAction); }
      catch (error) { return stop('illegal-action', { valid: false, error: error.message }); }
    }
    try {
      if (!Array.isArray(result.pv)) throw new Error('PV must be an array of complete actions.');
      if (result.pv.length && !isDeepStrictEqual(result.pv[0], result.bestAction)) throw new Error('PV does not start with the best action.');
      let pvPosition = current;
      for (const action of result.pv) pvPosition = validateAction(pvPosition, action);
    } catch (error) { return stop('invalid-pv', { valid: false, error: error.message }); }
    if (terminalClaim && next) return stop('false-terminal-claim', { valid: false,
      certificate: { terminal: false, verified: true, method: 'legal-action-witness', work: 0 } });
    // A validated submitted turn is already a full-rules witness that this
    // position is not terminal. Expensive unrestricted enumeration is needed
    // only when no action is available, for a terminal claim, or at the cap.
    if (!next || terminalClaim) {
      let certificate;
      try { certificate = certifyTerminal(current, limits); }
      catch (error) { return stop('terminal-verification-error', { valid: false, error: error.message }); }
      if (!certificate.verified) return stop(certificate.reason, { certificate });
      if (terminalClaim && (!certificate.terminal || certificate.status !== result.status)) {
        return stop('false-terminal-claim', { valid: false, certificate });
      }
      if (certificate.terminal) return finishTerminal(certificate, { lastSearch: search });
    }
    // Fixed-work benchmarks stop on a clock cutoff. The arena can use the
    // validated move retained by a time-limited search, just like self-play.
    if (result.stoppedReason === 'time' && (!playOnTimeLimit || !next)) return stop('time-limit');
    if (result.stoppedReason === 'policy') return stop('policy-boundary');
    if (!next) return stop(!result.completed ? 'incomplete-search' : 'missing-action');
    let notation;
    try { notation = formatAction(current, result.bestAction); }
    catch { notation = null; }
    game.moves.push({ ply, color, engine, beforeKey: positionKey(current), afterKey: positionKey(next),
      action: structuredClone(result.bestAction), notation, search });
    current = next;
  }
}

export function summarizeGames(games) {
  const valid = games.filter(game => game.valid), finished = valid.filter(game => game.result !== 'UNFINISHED');
  const aWins = finished.filter(game => game.result === 'A_WIN').length;
  const bWins = finished.filter(game => game.result === 'B_WIN').length;
  const draws = finished.filter(game => game.result === 'DRAW').length;
  return { games: games.length, finished: finished.length, unfinished: games.filter(game => game.result === 'UNFINISHED').length,
    invalid: games.filter(game => !game.valid).length, aWins, bWins, draws,
    incompleteSearchMoves: games.reduce((sum, game) => sum + (game.moves?.filter(move => !move.search.completed).length ?? 0), 0),
    aPoints: aWins + draws / 2, bPoints: bWins + draws / 2,
    aScoreAmongFinished: finished.length ? (aWins + draws / 2) / finished.length : null,
    unfinishedReasons: Object.fromEntries([...new Set(games.filter(game => game.result === 'UNFINISHED').map(game => game.reason))]
      .map(reason => [reason, games.filter(game => game.result === 'UNFINISHED' && game.reason === reason).length])) };
}

/** Completed pairs keep both color assignments in the scoring denominator. */
export function summarizePairs(pairs) {
  const complete = pairs.filter(pair => pair.complete);
  const aPoints = complete.reduce((sum, pair) => sum + pair.summary.aPoints, 0);
  const bPoints = complete.reduce((sum, pair) => sum + pair.summary.bPoints, 0);
  return { pairs: complete.length, aPoints, bPoints,
    aScore: complete.length ? aPoints / (2 * complete.length) : null,
    aBetter: complete.filter(pair => pair.summary.aPoints > pair.summary.bPoints).length,
    bBetter: complete.filter(pair => pair.summary.bPoints > pair.summary.aPoints).length,
    tied: complete.filter(pair => pair.summary.aPoints === pair.summary.bPoints).length };
}

export async function loadMatchSuite(suitePath = defaultSuite) {
  const url = suitePath instanceof URL ? suitePath : pathToFileURL(resolve(suitePath));
  const suite = JSON.parse(await readFile(url, 'utf8'));
  if (!Array.isArray(suite.cases) || !suite.cases.length) throw new Error('Suite needs nonempty cases.');
  const ids = new Set();
  const cases = [];
  for (const fixture of suite.cases) {
    if (typeof fixture.id !== 'string' || ids.has(fixture.id)) throw new Error('Each case needs a unique string id.');
    ids.add(fixture.id);
    const pgn = fixture.file ? await readFile(new URL(fixture.file, url), 'utf8') : fixture.pgn;
    cases.push({ ...fixture, position: fixture.position ? structuredClone(fixture.position) : createPosition({ variant: fixture.variant, pgn }) });
  }
  return { ...suite, cases };
}

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => { value = (Math.imul(value, 1664525) + 1013904223) >>> 0; return value / 4294967296; };
}

export async function runMatchSuite({ engineA = analyze, engineB = analyze, suite, suitePath,
  caseIds, seeds = [0], openingPlies = 0, onGame, ...options } = {}) {
  const limits = limitsFor(options);
  suite ??= await loadMatchSuite(suitePath);
  const cases = caseIds ? suite.cases.filter(fixture => caseIds.includes(fixture.id)) : suite.cases;
  if (!cases.length || caseIds?.some(id => !cases.some(fixture => fixture.id === id))) throw new Error('Unknown or empty match case selection.');
  if (!seeds.length || seeds.some(seed => !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)) throw new Error('Seeds must be unsigned 32-bit integers.');
  if (!Number.isInteger(openingPlies) || openingPlies < 0 || openingPlies > 100) throw new Error('Invalid openingPlies.');
  const games = [], pairs = [];
  for (const seed of seeds) for (const fixture of cases) {
    const random = seededRandom(seed);
    let position = fixture.position ? structuredClone(fixture.position) : createPosition({ pgn: fixture.pgn, variant: fixture.variant });
    const opening = [];
    for (let ply = 0; ply < openingPlies; ply++) {
      // A bounded deterministic prefix avoids enumerating enormous multiboard action sets.
      const choices = [];
      let work = 0;
      const exhausted = new Error('Opening generation work limit.');
      try {
        for (const candidate of generateActions(position, {
          pruneUnsafe: false, cacheMoves: false,
          tick() { if (++work > limits.terminalWork) throw exhausted; },
        })) { choices.push(candidate); if (choices.length === 64) break; }
      } catch (error) { if (error !== exhausted) throw error; }
      if (!choices.length) break;
      const selected = choices[Math.floor(random() * choices.length)];
      opening.push(structuredClone(selected.moves));
      position = validateAction(position, selected.moves);
    }
    const pairGames = [];
    for (const aColor of [0, 1]) {
      const game = { caseId: fixture.id, category: fixture.category ?? null, seed, opening,
        ...await runGame({ position, engineA, engineB, aColor, ...limits }) };
      games.push(game); pairGames.push(game);
      if (onGame) await onGame(game);
    }
    const pairSummary = summarizeGames(pairGames);
    const deterministicSearch = ({ elapsedMs, ...search }) => search;
    const trace = game => JSON.stringify({
      moves: game.moves.map(move => ({ action: move.action, search: deterministicSearch(move.search) })),
      lastSearch: game.lastSearch ? deterministicSearch(game.lastSearch) : null,
    });
    pairs.push({ caseId: fixture.id, seed, initialKey: positionKey(position),
      complete: pairSummary.finished === 2, summary: pairSummary,
      selfMatchConsistent: engineA === engineB ? trace(pairGames[0]) === trace(pairGames[1])
        && pairGames[0].reason === pairGames[1].reason && pairGames[0].winnerColor === pairGames[1].winnerColor : null });
  }
  return { suiteVersion: suite.version ?? null, description: suite.description ?? '', limits, seeds, openingPlies,
    methodology: 'Equal deterministic work budgets; each position is played with engine colors swapped. Only independently certified checkmate or stalemate earns points. Unfinished games earn no points. Completed-pair scores retain both colors; scores among finished games alone can be biased by unequal unfinished rates. Seeds without opening plies repeat the same positions and are not independent samples. No Elo estimate, repetition draw or evaluation-based adjudication.',
    environment: { node: process.version, platform: process.platform, arch: process.arch },
    summary: { ...summarizeGames(games), completePairs: pairs.filter(pair => pair.complete).length,
      totalPairs: pairs.length, uniqueStartingPositions: new Set(pairs.map(pair => pair.initialKey)).size,
      completedPairScore: summarizePairs(pairs),
      inconsistentSelfPairs: pairs.filter(pair => pair.selfMatchConsistent === false).length }, pairs, games };
}

const help = `Usage: node scripts/match.js [options]
  --engine-a FILE   Search module exporting analyze (default src/search.js)
  --engine-b FILE   Opponent module exporting analyze (default src/search.js)
  --suite FILE      JSON suite with cases containing id and pgn/file/position
  --case ID,ID      Restrict cases
  --seed N,N        Reproducible opening seeds (default 0)
  --opening-plies N Apply seeded legal opening turns to each case (default 0)
  --nodes N         Equal per-turn work budget (default 10000)
  --plies N         Maximum played turns per game (default 40)
  --depth N         Search depth (default 3)
  --qdepth N        Quiescence depth (default 1)
  --time-ms N       Safety cap for each search (default 30000)
  --terminal-time-ms N Independent terminal-check time cap (defaults to --time-ms)
  --terminal-work N Independent legal-turn verification cap (default 20000)
  --output FILE     Write full JSON report including move traces
  --json            Print full JSON report
  --help            Show this help`;

export function parseArguments(args) {
  const parsed = { options: {}, engineAPath: 'src/search.js', engineBPath: 'src/search.js' };
  const names = { '--nodes': 'maxNodes', '--plies': 'maxPlies', '--depth': 'maxDepth', '--qdepth': 'quiescenceDepth',
    '--time-ms': 'timeMs', '--terminal-time-ms': 'terminalTimeMs', '--terminal-work': 'terminalWork', '--opening-plies': 'openingPlies' };
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    if (flag === '--help') parsed.help = true;
    else if (flag === '--json') parsed.json = true;
    else {
      const value = args[++i];
      if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value.`);
      if (names[flag]) {
        if (!/^\d+$/.test(value)) throw new Error(`${flag} requires an integer.`);
        parsed.options[names[flag]] = Number(value);
      } else if (flag === '--engine-a') parsed.engineAPath = value;
      else if (flag === '--engine-b') parsed.engineBPath = value;
      else if (flag === '--suite') parsed.options.suitePath = value;
      else if (flag === '--case') parsed.options.caseIds = value.split(',');
      else if (flag === '--seed') {
        if (!/^\d+(,\d+)*$/.test(value)) throw new Error('--seed requires comma-separated integers.');
        parsed.options.seeds = [...new Set(value.split(',').map(Number))];
      } else if (flag === '--output') parsed.output = value;
      else throw new Error(`Unknown option: ${flag}`);
    }
  }
  return parsed;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const parsed = parseArguments(process.argv.slice(2));
    if (parsed.help) console.log(help);
    else {
      const engineA = (await import(pathToFileURL(resolve(parsed.engineAPath)))).analyze;
      const engineB = (await import(pathToFileURL(resolve(parsed.engineBPath)))).analyze;
      if (typeof engineA !== 'function' || typeof engineB !== 'function') throw new Error('Both engine modules must export analyze.');
      const report = await runMatchSuite({ ...parsed.options, engineA, engineB });
      report.engines = { A: parsed.engineAPath, B: parsed.engineBPath };
      if (parsed.output) {
        const target = resolve(parsed.output);
        await mkdir(dirname(target), { recursive: true });
        await writeFile(target, `${JSON.stringify(report, null, 2)}\n`);
      }
      if (parsed.json) console.log(JSON.stringify(report, null, 2));
      else {
        console.table(report.games.map(game => ({ case: game.caseId, seed: game.seed, white: game.white,
          result: game.result, reason: game.reason, plies: game.plies, valid: game.valid })));
        console.log(JSON.stringify(report.summary, null, 2));
        console.log(report.methodology);
      }
      if (report.summary.invalid || report.summary.inconsistentSelfPairs) process.exitCode = 1;
    }
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
