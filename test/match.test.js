import test from 'node:test';
import assert from 'node:assert/strict';
import { analyze } from '../src/search.js';
import { createPosition, generateActions, positionKey, validateAction } from '../src/rules.js';
import { certifyTerminal, loadMatchSuite, parseArguments, runGame, runMatchSuite, summarizeGames, summarizePairs } from '../scripts/match.js';

const tiny = () => createPosition({ pgn: '[Board "Custom"]\n[Size "4x4"]\n[2rk/4/4/KR2:0:1:w]' });
const mateInOne = () => createPosition({ pgn: '[Board "Custom"]\n[Size "4x4"]\n[Promotions "Q,R,B,N"]\n[3k/1P2/4/K3:0:1:w]' });
const options = { maxNodes: 1000, maxDepth: 1, quiescenceDepth: 0, timeMs: 10000, maxPlies: 2 };
function firstLegal(position, overrides = {}) {
  const iterator = generateActions(position);
  let action;
  try { action = iterator.next().value?.moves ?? null; }
  finally { iterator.return(); }
  return { bestAction: action, pv: action ? [action] : [], status: 'ok', completed: true,
    nodes: 2, searchNodes: 1, generationNodes: 1, score: 99999, stoppedReason: 'depth', ...overrides };
}

test('zero depth remains invalid for default and classical match callers', async () => {
  for (const engine of [undefined, 'classical']) {
    await assert.rejects(runGame({ position: tiny(), ...options, engine, maxDepth: 0 }), /Invalid maxDepth/);
  }
  const parsed = parseArguments(['--depth', '0']);
  await assert.rejects(runMatchSuite({ ...parsed.options,
    suite: { cases: [{ id: 'tiny', position: tiny() }] },
  }), /Invalid maxDepth/);
});

test('default match fixtures are legal nonterminal starts separate from tactics', async () => {
  const suite = await loadMatchSuite();
  assert(suite.cases.length >= 8);
  for (const fixture of suite.cases) {
    assert.equal(certifyTerminal(fixture.position).terminal, false, fixture.id);
    assert.equal(certifyTerminal(fixture.position).verified, true, fixture.id);
  }
});

test('real self match has identical legal traces and balanced paired mate results', async () => {
  const position = mateInOne(), original = structuredClone(position);
  const report = await runMatchSuite({ engineA: analyze, engineB: analyze,
    suite: { cases: [{ id: 'promotion', position }] }, ...options, maxPlies: 1 });
  assert.deepEqual(position, original);
  assert.equal(report.summary.completePairs, 1);
  assert.equal(report.summary.aWins, 1);
  assert.equal(report.summary.bWins, 1);
  assert.equal(report.summary.draws, 0);
  assert.equal(report.pairs[0].selfMatchConsistent, true);
  for (const game of report.games) {
    assert.equal(game.reason, 'checkmate');
    assert.equal(game.certificate.verified, true);
    assert.equal(game.plies, 1, 'The final allowed ply must still be checked for mate.');
    let replay = structuredClone(game.initialPosition);
    for (const row of game.moves) {
      assert.equal(row.beforeKey, positionKey(replay));
      replay = validateAction(replay, row.action);
      assert.equal(row.afterKey, positionKey(replay));
      assert(row.search.nodes <= options.maxNodes);
    }
    assert.equal(positionKey(replay), game.finalKey);
    assert(certifyTerminal(replay).terminal);
  }
});

test('equal limits and opposite assignments are used in both games', async () => {
  const calls = [];
  const engine = name => (position, limits) => { calls.push({ name, color: position.action % 2, limits }); return firstLegal(position); };
  const seen = [];
  const report = await runMatchSuite({ engineA: engine('A'), engineB: engine('B'),
    suite: { cases: [{ id: 'mini', position: tiny() }] }, ...options, onGame: game => seen.push(game) });
  assert.equal(calls.length, 4);
  assert.deepEqual(calls.map(call => [call.name, call.color]), [['A', 0], ['B', 1], ['B', 0], ['A', 1]]);
  assert(calls.every(call => JSON.stringify(call.limits) === JSON.stringify(calls[0].limits)));
  assert.equal(calls[0].limits.maxNodes, options.maxNodes);
  assert.equal(seen.length, 2);
  assert.equal(report.summary.unfinished, 2);
  assert.equal(report.summary.aPoints, 0);
  assert.equal(report.summary.bPoints, 0);
});

test('a nondeterministic self engine is exposed by the paired consistency check', async () => {
  let calls = 0;
  const engine = position => {
    const result = firstLegal(position);
    if (++calls % 2 === 0) {
      const iterator = generateActions(position);
      try { iterator.next(); result.bestAction = iterator.next().value.moves; }
      finally { iterator.return(); }
      result.pv = [result.bestAction];
    }
    return result;
  };
  const report = await runMatchSuite({ engineA: engine, engineB: engine,
    suite: { cases: [{ id: 'mini', position: tiny() }] }, ...options, maxPlies: 1 });
  assert.equal(report.pairs[0].selfMatchConsistent, false);
  assert.equal(report.summary.inconsistentSelfPairs, 1);
});

test('high evaluation at a ply cap never adjudicates a win or draw', async () => {
  const game = await runGame({ position: tiny(), engineA: firstLegal, engineB: firstLegal, ...options, maxPlies: 1 });
  assert.equal(game.result, 'UNFINISHED');
  assert.equal(game.reason, 'ply-limit');
  assert.equal(game.moves[0].search.score, 99999);
  assert.equal(game.winner, null);
});

test('incomplete, absent, policy and time-limited results never forfeit', async () => {
  for (const [overrides, reason] of [
    [{ completed: false, status: 'incomplete', stoppedReason: 'nodes', bestAction: null, pv: [] }, 'incomplete-search'],
    [{ bestAction: null, pv: [] }, 'missing-action'],
    [{ bestAction: null, pv: [], stoppedReason: 'policy' }, 'policy-boundary'],
    [{ stoppedReason: 'time' }, 'time-limit'],
  ]) {
    const game = await runGame({ position: tiny(), ...options,
      engineA: position => firstLegal(position, overrides), engineB: firstLegal });
    assert.equal(game.result, 'UNFINISHED');
    assert.equal(game.reason, reason);
    assert.equal(game.plies, 0);
    assert.equal(game.winner, null);
    assert.equal(game.valid, true);
  }
});

test('a completed iteration may play on after its fixed node budget is reached', async () => {
  const game = await runGame({ position: tiny(), ...options, maxPlies: 1,
    engineA: position => firstLegal(position, { stoppedReason: 'nodes' }), engineB: firstLegal });
  assert.equal(game.plies, 1);
  assert.equal(game.reason, 'ply-limit');
});

test('a validated incomplete fallback remains playable and is counted in the report', async () => {
  const engine = position => firstLegal(position, { completed: false, status: 'incomplete', stoppedReason: 'nodes', score: null });
  const game = await runGame({ position: tiny(), ...options, maxPlies: 1, engineA: engine, engineB: engine });
  assert.equal(game.plies, 1);
  assert.equal(game.reason, 'ply-limit');
  assert.equal(game.result, 'UNFINISHED');
  assert.equal(game.moves[0].search.completed, false);
  assert.equal(summarizeGames([game]).incompleteSearchMoves, 1);
});

test('transformer diagnostics survive both played fallbacks and stopped searches', async () => {
  const diagnostics = { engine: 'transformer', searchingDepth: 1, rootActionsSearched: 0,
    selectiveDepth: 1, candidateLimit: 64, innerCandidateLimit: 16, candidateCaps: 1,
    candidateCacheEntries: 0, evaluations: 12, inferenceBatches: 1, mateProven: false, terminalProof: null };
  const engine = position => firstLegal(position, { ...diagnostics,
    completed: false, status: 'incomplete', stoppedReason: 'time', score: 42, depth: 0 });
  for (const playOnTimeLimit of [true, false]) {
    const game = await runGame({ position: tiny(), ...options, maxPlies: 1,
      playOnTimeLimit, engineA: engine, engineB: engine });
    assert.equal(game.valid, true);
    assert.equal(game.reason, playOnTimeLimit ? 'ply-limit' : 'time-limit');
    assert.equal(game.plies, playOnTimeLimit ? 1 : 0);
    const search = playOnTimeLimit ? game.moves[0].search : game.lastSearch;
    for (const [key, value] of Object.entries(diagnostics)) {
      assert(Object.hasOwn(search, key), `Missing recorded diagnostic: ${key}`);
      assert.deepEqual(search[key], value, key);
    }
  }
});

test('classical searches retain depth diagnostics without inventing transformer fields', async () => {
  let result;
  const engine = position => { result = analyze(position, options); return result; };
  const game = await runGame({ position: tiny(), ...options, maxPlies: 1, engineA: engine, engineB: engine });
  assert.equal(game.valid, true);
  assert.equal(game.plies, 1);
  const search = game.moves[0].search;
  for (const key of ['searchingDepth', 'rootActionsSearched', 'selectiveDepth']) {
    assert.equal(search[key], result[key], key);
  }
  for (const key of ['engine', 'candidateLimit', 'innerCandidateLimit', 'candidateCaps',
    'candidateCacheEntries', 'evaluations', 'inferenceBatches', 'mateProven', 'terminalProof']) {
    assert.equal(Object.hasOwn(search, key), false, key);
  }
});

test('time-budget play is explicit and does not forgive absent actions or policy stops', async () => {
  for (const completed of [true, false]) {
    const engine = position => firstLegal(position, { completed, status: completed ? 'ok' : 'incomplete',
      stoppedReason: 'time', score: completed ? 42 : null });
    const game = await runGame({ position: tiny(), ...options, maxPlies: 1,
      playOnTimeLimit: true, engineA: engine, engineB: engine });
    assert.equal(game.playOnTimeLimit, true);
    assert.equal(game.plies, 1);
    assert.equal(game.reason, 'ply-limit');
    assert.equal(game.result, 'UNFINISHED');
    assert.equal(game.moves[0].search.completed, completed);
    assert.equal(game.moves[0].search.stoppedReason, 'time');
    assert.equal(game.moves[0].search.score, completed ? 42 : null);
    assert.equal(positionKey(validateAction(game.initialPosition, game.moves[0].action)), game.finalKey);
    assert.equal(summarizeGames([game]).incompleteSearchMoves, completed ? 0 : 1);
  }
  for (const [overrides, reason] of [
    [{ stoppedReason: 'time', bestAction: null, pv: [], completed: false }, 'time-limit'],
    [{ stoppedReason: 'policy' }, 'policy-boundary'],
  ]) {
    const game = await runGame({ position: tiny(), ...options, playOnTimeLimit: true,
      engineA: position => firstLegal(position, overrides), engineB: firstLegal });
    assert.equal(game.plies, 0);
    assert.equal(game.reason, reason);
    assert.equal(game.result, 'UNFINISHED');
    assert.equal(game.valid, true);
  }
  await assert.rejects(runGame({ position: tiny(), playOnTimeLimit: 'true' }), /playOnTimeLimit/);
});

test('illegal later principal-variation actions invalidate a search result', async () => {
  const engineA = position => {
    const result = firstLegal(position);
    result.pv.push([[[0, 0, 0, 0], [0, 0, 3, 3]]]);
    return result;
  };
  const game = await runGame({ position: tiny(), ...options, engineA, engineB: firstLegal });
  assert.equal(game.reason, 'invalid-pv');
  assert.equal(game.valid, false);
  assert.equal(game.result, 'UNFINISHED');
});

test('false terminal claims, illegal actions, mutation and bad counters invalidate without awarding points', async () => {
  const badEngines = [
    [position => firstLegal(position, { status: 'checkmate', bestAction: null, pv: [] }), 'false-terminal-claim'],
    [position => firstLegal(position, { bestAction: [[[0, 0, 0, 0], [0, 0, 3, 3]]] }), 'illegal-action'],
    [position => { position.promotions.reverse(); return firstLegal(position); }, 'input-mutation'],
    [position => firstLegal(position, { nodes: 1001 }), 'invalid-work-accounting'],
    [position => firstLegal(position, { searchNodes: -1, generationNodes: 3 }), 'invalid-work-accounting'],
    [position => firstLegal(position, { bestAction: () => null }), 'invalid-result'],
  ];
  for (const [engineA, reason] of badEngines) {
    const position = tiny(), before = structuredClone(position);
    const game = await runGame({ position, engineA, engineB: firstLegal, ...options });
    assert.deepEqual(position, before);
    assert.equal(game.valid, false);
    assert.equal(game.reason, reason);
    assert.equal(game.result, 'UNFINISHED');
    assert.equal(game.winner, null);
  }
});

test('a legal turn disproves a terminal claim even with no enumeration budget', async () => {
  const game = await runGame({ position: tiny(), ...options, terminalWork: 0,
    engineA: position => firstLegal(position, { status: 'checkmate' }), engineB: firstLegal });
  assert.equal(game.reason, 'false-terminal-claim');
  assert.equal(game.valid, false);
  assert.equal(game.result, 'UNFINISHED');
  assert.equal(game.certificate.method, 'legal-action-witness');
});

test('only exhaustive checkmate or stalemate certificates count, and bounded verification stays unfinished', async () => {
  const terminal = createPosition({ pgn: '[Board "Custom"]\n[Size "3x3"]\n[1q1/2k/K2:0:1:w]' });
  const engine = () => { throw new Error('Terminal positions must not ask an engine to move.'); };
  const stalemate = await runGame({ position: terminal, engineA: engine, engineB: engine, ...options, maxPlies: 0 });
  assert.equal(stalemate.result, 'DRAW');
  assert.equal(stalemate.reason, 'stalemate');
  assert.equal(stalemate.certificate.inCheck, false);
  const bounded = await runGame({ position: terminal, engineA: engine, engineB: engine, ...options, maxPlies: 0, terminalWork: 0 });
  assert.equal(bounded.result, 'UNFINISHED');
  assert.equal(bounded.reason, 'terminal-work-limit');
  assert.equal(bounded.certificate.verified, false);
});

test('a legal engine turn bypasses redundant terminal enumeration before play', async () => {
  const game = await runGame({ position: tiny(), ...options, engineA: firstLegal, engineB: firstLegal,
    terminalWork: 0, maxPlies: 1 });
  assert.equal(game.plies, 1, 'A validated legal turn witnesses nonterminal play despite a zero certification budget.');
  assert.equal(game.reason, 'terminal-work-limit');
  assert.equal(game.result, 'UNFINISHED');
});

test('no-action responses are independently classified and wrong terminal kinds are rejected', async () => {
  const position = createPosition({ pgn: '[Board "Custom"]\n[Size "3x3"]\n[1q1/2k/K2:0:1:w]' });
  const empty = overrides => () => ({ bestAction: null, pv: [], nodes: 0, searchNodes: 0, generationNodes: 0,
    status: 'incomplete', completed: false, stoppedReason: 'nodes', ...overrides });
  const verified = await runGame({ position, ...options, engineA: empty(), engineB: empty() });
  assert.equal(verified.result, 'DRAW');
  assert.equal(verified.reason, 'stalemate');
  const wrong = await runGame({ position, ...options, engineA: empty({ status: 'checkmate' }), engineB: empty() });
  assert.equal(wrong.reason, 'false-terminal-claim');
  assert.equal(wrong.valid, false);
  assert.equal(wrong.result, 'UNFINISHED');
});

test('aggregation excludes all unfinished and invalid results from points and denominators', () => {
  const report = summarizeGames([
    { valid: true, result: 'A_WIN' }, { valid: true, result: 'B_WIN' }, { valid: true, result: 'DRAW' },
    { valid: true, result: 'UNFINISHED', reason: 'ply-limit' },
    { valid: false, result: 'UNFINISHED', reason: 'illegal-action' },
    { valid: false, result: 'A_WIN' },
  ]);
  assert.equal(report.finished, 3);
  assert.equal(report.aPoints, 1.5);
  assert.equal(report.bPoints, 1.5);
  assert.equal(report.aScoreAmongFinished, 0.5);
  assert.equal(report.invalid, 2);
  assert.deepEqual(report.unfinishedReasons, { 'ply-limit': 1, 'illegal-action': 1 });
  assert.equal(summarizeGames([]).aScoreAmongFinished, null);
});

test('completed-pair scoring omits both games when either color assignment is unfinished', () => {
  const result = summarizePairs([
    { complete: true, summary: { aPoints: 2, bPoints: 0 } },
    { complete: true, summary: { aPoints: 1, bPoints: 1 } },
    { complete: true, summary: { aPoints: 0, bPoints: 2 } },
    { complete: false, summary: { aPoints: 1, bPoints: 0 } },
  ]);
  assert.deepEqual(result, { pairs: 3, aPoints: 3, bPoints: 3, aScore: 0.5, aBetter: 1, bBetter: 1, tied: 1 });
  assert.equal(summarizePairs([]).aScore, null);
});

test('terminal verification safety timeout remains unclassified', () => {
  const result = certifyTerminal(tiny(), { timeMs: 0 });
  assert.equal(result.verified, false);
  assert.equal(result.terminal, false);
  assert.equal(result.reason, 'terminal-time-limit');
});

test('terminal verification time budget is independent and falls back to the search cap for old callers', t => {
  let clock = 0;
  t.mock.method(performance, 'now', () => clock++);
  const inherited = certifyTerminal(tiny(), { timeMs: 1 });
  assert.equal(inherited.verified, false);
  assert.equal(inherited.reason, 'terminal-time-limit');
  const extended = certifyTerminal(tiny(), { timeMs: 1, terminalTimeMs: 100 });
  assert.equal(extended.verified, true);
  assert.equal(extended.terminal, false);
  const independent = certifyTerminal(tiny(), { timeMs: 100, terminalTimeMs: 1 });
  assert.equal(independent.verified, false);
  assert.equal(independent.reason, 'terminal-time-limit');
});

test('seeded opening variations replay legally and preserve paired starts', async () => {
  const args = { engineA: firstLegal, engineB: firstLegal, suite: { cases: [{ id: 'mini', position: tiny() }] },
    ...options, seeds: [1, 7], openingPlies: 2, maxPlies: 0 };
  const first = await runMatchSuite(args), repeat = await runMatchSuite(args);
  assert.deepEqual(first, repeat);
  for (let i = 0; i < first.games.length; i += 2) {
    const game = first.games[i];
    assert.equal(game.initialKey, first.games[i + 1].initialKey);
    let start = tiny();
    for (const action of game.opening) start = validateAction(start, action);
    assert.equal(game.initialKey, positionKey(start));
  }
});

test('case selection and CLI values reject mistakes before running trials', async () => {
  assert.deepEqual(parseArguments(['--time-ms', '600', '--terminal-time-ms', '3000']).options,
    { timeMs: 600, terminalTimeMs: 3000 });
  assert.deepEqual(parseArguments(['--nodes', '42', '--case', 'one,two', '--seed', '2,3', '--engine-a', 'old.js']).options,
    { maxNodes: 42, caseIds: ['one', 'two'], seeds: [2, 3] });
  assert.throws(() => parseArguments(['--nodes', '-2']));
  assert.throws(() => parseArguments(['--seed', '1,no']));
  await assert.rejects(runMatchSuite({ suite: { cases: [{ id: 'mini', position: tiny() }] }, caseIds: ['absent'] }));
  await assert.rejects(runMatchSuite({ maxNodes: -1 }));
  await assert.rejects(runMatchSuite({ terminalTimeMs: 0 }));
});
