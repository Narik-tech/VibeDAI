import test from 'node:test';
import assert from 'node:assert/strict';
import { createPosition, generateActions, positionKey, validateAction } from '../src/rules.js';
import { analyze } from '../src/search.js';
import { analyze as analyzeTransformer } from '../src/transformer-search.js';
import { decidePromotion, evaluateCandidate } from '../scripts/transformer-selfplay-arena.js';
import { certifyTerminal, loadMatchSuite } from '../scripts/match.js';

const tiny = () => createPosition({ pgn: '[Board "Custom"]\n[Size "4x4"]\n[2rk/4/4/KR2:0:1:w]' });
const mating = () => createPosition({ pgn: '[Board "Custom"]\n[Size "4x4"]\n[Promotions "Q,R,B,N"]\n[3k/1P2/4/K3:0:1:w]' });
const terminal = () => createPosition({ pgn: '[Board "Custom"]\n[Size "3x3"]\n[1q1/2k/K2:0:1:w]' });
const limits = { maxPlies: 1, maxNodes: 1000, maxDepth: 1, timeMs: 10000 };

function firstLegal(position) {
  const iterator = generateActions(position);
  let bestAction;
  try { bestAction = iterator.next().value?.moves ?? null; }
  finally { iterator.return(); }
  return { bestAction, pv: bestAction ? [bestAction] : [], status: 'ok', completed: true,
    nodes: 2, searchNodes: 1, generationNodes: 1, stoppedReason: 'depth', score: 99999 };
}

const nextTurn = () => new Promise(resolve => setImmediate(resolve));
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}
const parallelSuite = () => ({ cases: [{ id: 'tiny', position: tiny() }, { id: 'mating', position: mating() }] });

test('dynamic depth reaches both arena engines and preserves requested game limits', async () => {
  const calls = [];
  const engine = name => (position, options) => {
    calls.push({ name, depth: options.maxDepth, engine: options.engine });
    return firstLegal(position);
  };
  const report = await evaluateCandidate({ candidate: engine('candidate'), incumbent: engine('incumbent'),
    suite: { cases: [{ id: 'tiny', position: tiny() }] }, pairs: 1, minPairs: 1,
    ...limits, maxDepth: 0,
  });
  assert.deepEqual(calls, [
    { name: 'candidate', depth: 0, engine: 'transformer' },
    { name: 'incumbent', depth: 0, engine: 'transformer' },
  ]);
  assert.equal(report.limits.maxDepth, 0);
  assert(report.games.every(game => game.valid && game.limits.maxDepth === 0));
});

function syntheticPairs(outcomes, { duplicate = false, unplayed = false } = {}) {
  const games = [], pairs = [];
  for (const [index, results] of outcomes.entries()) {
    const initialKey = duplicate ? 'same' : `position-${index}`, gameIndices = [];
    for (const [aColor, result] of results.entries()) {
      const winnerColor = result === 'DRAW' ? null : result === 'A_WIN' ? aColor : 1 - aColor;
      const moves = unplayed ? [] : [{ engine: aColor === 0 ? 'A' : 'B', search: { completed: true } },
        { engine: aColor === 1 ? 'A' : 'B', search: { completed: true } }];
      gameIndices.push(games.length);
      games.push({ aColor, initialKey, result, valid: true, plies: moves.length, moves,
        certificate: { terminal: true, verified: true, winnerColor, status: result === 'DRAW' ? 'stalemate' : 'checkmate' } });
    }
    pairs.push({ initialKey, gameIndices });
  }
  return { games, pairs };
}

test('promotion requires a strict winning margin and enough distinct complete played pairs', () => {
  const winning = syntheticPairs(Array(4).fill(['A_WIN', 'A_WIN']));
  assert.equal(decidePromotion(winning).promote, true);
  assert.equal(decidePromotion(winning).candidateScore, 1);
  const tied = syntheticPairs(Array(4).fill(['A_WIN', 'B_WIN']));
  assert.equal(decidePromotion({ ...tied, promotionScore: 0.5 }).reason, 'no-winning-margin');
  const duplicate = syntheticPairs(Array(4).fill(['A_WIN', 'A_WIN']), { duplicate: true });
  assert.equal(decidePromotion(duplicate).eligiblePairs, 1);
  assert.equal(decidePromotion(duplicate).promote, false);
  const unplayed = syntheticPairs(Array(4).fill(['A_WIN', 'A_WIN']), { unplayed: true });
  assert.equal(decidePromotion(unplayed).eligiblePairs, 0);
});

test('strength estimates use distinct color-swapped pairs and stay inconclusive for a small all-win sample', () => {
  const winning = syntheticPairs(Array(4).fill(['A_WIN', 'A_WIN']));
  const decision = decidePromotion(winning);
  assert.equal(decision.promote, true, 'the operational gate remains backwards compatible');
  const assessment = decision.strengthAssessment;
  assert.equal(assessment.status, 'inconclusive');
  assert.equal(assessment.distinctPairs, 4);
  assert(assessment.reasons.includes('insufficient-distinct-pairs'));
  const interval = assessment.confidenceInterval;
  assert.equal(interval.unit, 'distinct-color-swapped-pair');
  assert.equal(interval.level, 0.95);
  assert.equal(interval.estimate, 1);
  assert.equal(interval.upper, 1);
  assert(Math.abs(interval.lower - (1 - Math.sqrt(Math.log(40) / 8))) < 1e-12);
  assert(interval.lower < 0.5, 'eight correlated games must not masquerade as eight independent pairs');
  const duplicate = decidePromotion(syntheticPairs(Array(20).fill(['A_WIN', 'A_WIN']), { duplicate: true }));
  assert.equal(duplicate.strengthAssessment.distinctPairs, 1);
  assert.equal(duplicate.strengthAssessment.status, 'inconclusive');
});

test('larger paired reports distinguish advantage, equal scores and unfinished selection bias', () => {
  for (const [outcomes, status] of [
    [['A_WIN', 'A_WIN'], 'candidate-advantage'],
    [['B_WIN', 'B_WIN'], 'incumbent-advantage'],
    [['A_WIN', 'B_WIN'], 'inconclusive'],
  ]) {
    const report = decidePromotion(syntheticPairs(Array(20).fill(outcomes)));
    assert.equal(report.strengthAssessment.status, status);
  }
  const unfinished = syntheticPairs(Array(21).fill(['A_WIN', 'A_WIN']));
  Object.assign(unfinished.games.at(-1), { result: 'UNFINISHED', reason: 'ply-limit' });
  const decision = decidePromotion(unfinished);
  assert.equal(decision.promote, true);
  const assessment = decision.strengthAssessment;
  assert.equal(assessment.status, 'inconclusive');
  assert(assessment.reasons.includes('unfinished-or-excluded-pairs'));
  assert.equal(assessment.distinctPairs, 20);
  assert.equal(assessment.unscoredPairs, 1);
  assert.deepEqual(assessment.allPlannedPairScoreRange, { lower: 20 / 21, upper: 1 });
  assert.equal(decidePromotion().strengthAssessment.confidenceInterval.lower, null);
});

test('arena terminal verification uses its independent budget without increasing the search budget', async () => {
  const calls = [];
  const engine = (position, options) => { calls.push(options); return firstLegal(position); };
  const report = await evaluateCandidate({ candidate: engine, incumbent: engine,
    suite: { cases: [{ id: 'tiny', position: tiny() }] }, pairs: 1, minPairs: 1,
    ...limits, timeMs: 600, terminalTimeMs: 3000 });
  assert.equal(report.limits.timeMs, 600);
  assert.equal(report.limits.terminalTimeMs, 3000);
  assert(calls.every(options => options.timeMs === 600 && options.terminalTimeMs === 3000));
  assert(report.games.every(game => game.limits.terminalTimeMs === 3000));
});

test('arena and frozen final fixtures are distinct legal nonterminal histories with diverse positions', async () => {
  const [training, arena, final] = await Promise.all(['training', 'validation', 'transformer-test']
    .map(name => loadMatchSuite(new URL(`../examples/matches/${name}.json`, import.meta.url))));
  assert.equal(arena.purpose, 'repeated-selection');
  assert.equal(final.purpose, 'final-test');
  assert.equal(final.frozen, true);
  assert(arena.cases.length >= 20);
  assert(final.cases.length >= 20);
  const keys = new Set(training.cases.map(fixture => positionKey(fixture.position)));
  for (const suite of [arena, final]) {
    assert(new Set(suite.cases.map(fixture => fixture.category)).size >= 6);
    assert(suite.cases.some(fixture => fixture.position.board.filter(Boolean).length > 1));
    for (const fixture of suite.cases) {
      const key = positionKey(fixture.position);
      assert(!keys.has(key), `overlapping full-history start: ${fixture.id}`);
      keys.add(key);
      if (fixture.provenance) {
        let replay = createPosition({ pgn: fixture.provenance.initialPgn });
        assert(fixture.provenance.opening.length >= 3);
        for (const action of fixture.provenance.opening) replay = validateAction(replay, action);
        assert.equal(positionKey(replay), key, `invalid opening history: ${fixture.id}`);
      } else assert(suite === arena, 'every frozen final fixture needs independently replayable provenance');
      const certificate = certifyTerminal(fixture.position, { terminalWork: 50000, terminalTimeMs: 10000 });
      assert.equal(certificate.verified, true, fixture.id);
      assert.equal(certificate.terminal, false, fixture.id);
      const witness = generateActions(fixture.position, { firstOnly: true });
      try { assert.equal(positionKey(validateAction(fixture.position, witness.next().value.moves)),
        positionKey(validateAction(fixture.position, firstLegal(fixture.position).bestAction))); }
      finally { witness.return(); }
    }
  }
});

test('unfinished pairs never score and any invalid game vetoes otherwise winning promotion', () => {
  const report = syntheticPairs(Array(5).fill(['A_WIN', 'A_WIN']));
  report.games.at(-1).result = 'UNFINISHED';
  assert.equal(decidePromotion(report).eligiblePairs, 4);
  assert.equal(decidePromotion(report).candidatePoints, 8);
  report.games.at(-1).valid = false;
  assert.equal(decidePromotion(report).reason, 'invalid-games');
  report.games.at(-1).valid = true;
  report.games[0].certificate.verified = false;
  assert.equal(decidePromotion(report).eligiblePairs, 3);
  assert.equal(decidePromotion(report).promote, false);
});

test('completion rates include unfinished games and distinguish certified, complete and eligible results', () => {
  const report = syntheticPairs(Array(5).fill(['A_WIN', 'A_WIN']));
  Object.assign(report.games[3], { result: 'UNFINISHED', reason: 'ply-limit' });
  for (const index of report.pairs[2].gameIndices) Object.assign(report.games[index], { moves: [], plies: 0 });
  report.pairs[3].initialKey = report.pairs[0].initialKey;
  for (const index of report.pairs[3].gameIndices) report.games[index].initialKey = report.pairs[0].initialKey;
  Object.assign(report.games[9], { result: 'UNFINISHED', valid: false, reason: 'engine-error' });

  const decision = decidePromotion({ ...report, minPairs: 1 });
  assert.equal(decision.reason, 'invalid-games');
  assert.equal(decision.candidateScore, 1);
  assert.equal(decision.candidatePoints, 2);
  assert.deepEqual(decision.completion, {
    totalGames: 10, certifiedGames: 8, gameCompletionRate: 0.8,
    unfinishedReasons: { 'ply-limit': 1, 'engine-error': 1 },
    totalPairs: 5, completePairs: 3, pairCompletionRate: 0.6,
    eligiblePairs: 1, eligiblePairRate: 0.2,
    byCandidateColor: {
      white: { totalGames: 5, certifiedGames: 5, gameCompletionRate: 1, unfinishedReasons: {} },
      black: { totalGames: 5, certifiedGames: 3, gameCompletionRate: 0.6,
        unfinishedReasons: { 'ply-limit': 1, 'engine-error': 1 } },
    },
  });
});

test('gate rejects mismatched starts, wrong color pairs, fake terminal results and below-threshold scores', () => {
  const report = syntheticPairs([['A_WIN', 'DRAW'], ['A_WIN', 'B_WIN']]);
  assert.equal(decidePromotion({ ...report, minPairs: 2, promotionScore: 0.7 }).reason, 'below-promotion-score');
  report.games[0].initialKey = 'different';
  assert.equal(decidePromotion({ ...report, minPairs: 1 }).eligiblePairs, 1);
  report.games[2].aColor = 1;
  assert.equal(decidePromotion({ ...report, minPairs: 1 }).eligiblePairs, 0);
  const fake = syntheticPairs([['A_WIN', 'A_WIN']]);
  fake.games[0].certificate.winnerColor = 1;
  assert.equal(decidePromotion({ ...fake, minPairs: 1 }).eligiblePairs, 0);
  assert.equal(decidePromotion({ ...fake, minPairs: 1 }).completion.certifiedGames, 1);
  assert.equal(decidePromotion({ ...fake, minPairs: 1 }).completion.gameCompletionRate, 0.5);
});

test('arena rotates deterministically, deduplicates complete positions, swaps colors and leaves inputs untouched', async () => {
  const suite = { cases: [{ id: 'tiny', position: tiny() }, { id: 'duplicate', position: tiny() }, { id: 'mating', position: mating() }] };
  const original = structuredClone(suite), calls = [], observed = [];
  const engine = name => async (position, options) => {
    calls.push({ name, color: position.action % 2, options });
    return firstLegal(position);
  };
  const report = await evaluateCandidate({ candidate: engine('A'), incumbent: engine('B'), suite, pairs: 4,
    minPairs: 4, seed: 1, ...limits, onGame: game => { observed.push(game.aColor); game.valid = false; } });
  assert.deepEqual(suite, original);
  assert.deepEqual(report.scheduledCases, ['duplicate', 'mating']);
  assert.deepEqual(report.skippedCases.map(item => item.reason), ['duplicate-position']);
  assert.deepEqual(observed, [0, 1, 0, 1]);
  assert.deepEqual(calls.map(call => call.name), ['A', 'B', 'A', 'B']);
  assert(calls.every(call => call.options.engine === 'transformer' && typeof call.options.shouldStop === 'function'));
  assert(report.games.every(game => game.valid));
  assert(report.pairs.every(pair => report.games[pair.gameIndices[0]].initialKey === report.games[pair.gameIndices[1]].initialKey));
  assert.equal(report.decision.promote, false);
  assert.deepEqual(report.summary.completion, report.decision.completion);
  assert.equal(report.summary.completion.totalPairs, 2);
  assert.equal(report.summary.completion.totalGames, 4);
});

test('parallel arena bounds overlapping games, serializes completion callbacks and preserves pair indices', { timeout: 5000 }, async () => {
  const firstGame = deferred(), observed = [];
  let launched = 0, active = 0, peak = 0, writing = 0, peakWriting = 0;
  const engine = async position => {
    const index = launched++;
    peak = Math.max(peak, ++active);
    try {
      if (index === 0) await firstGame.promise;
      return firstLegal(position);
    } finally { active--; }
  };
  const report = await evaluateCandidate({ candidate: engine, incumbent: engine, suite: parallelSuite(),
    pairs: 2, minPairs: 1, seed: 0, gameConcurrency: 2, ...limits,
    onGame: async (game, progress) => {
      assert.equal(game.index, progress.index);
      peakWriting = Math.max(peakWriting, ++writing);
      await nextTurn();
      observed.push({ index: game.index, ...progress });
      if (progress.index === 2) firstGame.resolve();
      game.valid = false;
      writing--;
    },
  });
  assert.equal(peak, 2);
  assert.equal(active, 0);
  assert.equal(peakWriting, 1, 'Completion callbacks must not overlap.');
  assert.deepEqual(observed.slice(0, 2).map(item => item.index), [1, 2]);
  assert.deepEqual(observed.map(item => item.completed), [1, 2, 3, 4]);
  assert(observed.every(item => item.total === 4));
  assert.deepEqual(observed.map(item => item.index).toSorted(), [0, 1, 2, 3]);
  assert.deepEqual(report.games.map(game => game.index), [0, 1, 2, 3]);
  assert.deepEqual(report.games.map(game => [game.caseId, game.aColor]),
    [['tiny', 0], ['tiny', 1], ['mating', 0], ['mating', 1]]);
  assert.deepEqual(report.pairs.map(pair => pair.gameIndices), [[0, 1], [2, 3]]);
  assert(report.games.every(game => game.valid));
  for (const pair of report.pairs) {
    assert(pair.gameIndices.every(index => report.games[index].initialKey === pair.initialKey));
  }
});

test('parallel and default serial arenas preserve deterministic selection, games and promotion decisions', async () => {
  const suite = { cases: [{ id: 'tiny', position: tiny() }, { id: 'duplicate', position: tiny() },
    { id: 'terminal', position: terminal() }, { id: 'mating', position: mating() }] };
  const original = structuredClone(suite);
  const engine = async (position, options) => {
    await nextTurn();
    const { elapsedMs, ...result } = analyze(position, options);
    return result;
  };
  const options = { candidate: engine, incumbent: engine, suite, pairs: 4, minPairs: 1, seed: 1, ...limits };
  const serial = await evaluateCandidate(options);
  assert.equal(serial.gameConcurrency, 1);
  assert.deepEqual(serial.scheduledCases, ['duplicate', 'mating']);
  assert.deepEqual(serial.skippedCases.map(item => item.reason), ['terminal-start', 'duplicate-position']);
  assert.equal(serial.decision.eligiblePairs, 1);
  assert.equal(serial.decision.candidateScore, 0.5);
  assert.equal(serial.decision.reason, 'no-winning-margin');
  for (const gameConcurrency of [1, 3, 8]) {
    const parallel = await evaluateCandidate({ ...options, gameConcurrency });
    assert.equal(parallel.gameConcurrency, gameConcurrency);
    assert.deepEqual({ ...parallel, gameConcurrency: 1 }, serial);
  }
  assert.deepEqual(suite, original);
});

test('arena plans and validates all scheduled cases before starting parallel engines', async () => {
  let calls = 0;
  const engine = position => { calls++; return firstLegal(position); };
  await assert.rejects(evaluateCandidate({ candidate: engine, incumbent: engine,
    suite: { cases: [{ id: 'tiny', position: tiny() }, { position: mating() }] },
    pairs: 2, seed: 0, gameConcurrency: 2, ...limits }), /string id/);
  assert.equal(calls, 0);
});

test('arena yields during planning so queued cancellation prevents game dispatch', async () => {
  let stop = false, calls = 0;
  const timer = setImmediate(() => { stop = true; });
  const engine = position => { calls++; return firstLegal(position); };
  try {
    await assert.rejects(evaluateCandidate({ candidate: engine, incumbent: engine,
      suite: parallelSuite(), pairs: 2, seed: 0, gameConcurrency: 2, ...limits,
      shouldStop: () => stop,
    }), { name: 'AbortError' });
    assert.equal(calls, 0);
  } finally { clearImmediate(timer); }
});

test('terminal starts never run engines or count towards a minimum', async () => {
  const unused = async () => { throw new Error('should not run'); };
  const report = await evaluateCandidate({ candidate: unused, incumbent: unused,
    suite: { cases: [{ id: 'terminal', position: terminal() }] }, pairs: 1, minPairs: 1, ...limits });
  assert.equal(report.games.length, 0);
  assert.equal(report.skippedCases[0].reason, 'terminal-start');
  assert.equal(report.decision.promote, false);
  assert.deepEqual(report.decision.completion, {
    totalGames: 0, certifiedGames: 0, gameCompletionRate: null, unfinishedReasons: {},
    totalPairs: 0, completePairs: 0, pairCompletionRate: null, eligiblePairs: 0, eligiblePairRate: null,
    byCandidateColor: {
      white: { totalGames: 0, certifiedGames: 0, gameCompletionRate: null, unfinishedReasons: {} },
      black: { totalGames: 0, certifiedGames: 0, gameCompletionRate: null, unfinishedReasons: {} },
    },
  });
});

test('real one-turn mates complete a balanced meaningful pair and do not promote a tie', async () => {
  const engine = async (position, options) => analyze(position, options);
  const report = await evaluateCandidate({ candidate: engine, incumbent: engine,
    suite: { cases: [{ id: 'mate', position: mating() }] }, pairs: 1, minPairs: 1, ...limits });
  assert.equal(report.decision.eligiblePairs, 1);
  assert.equal(report.decision.candidateScore, 0.5);
  assert.equal(report.decision.promote, false);
  assert(report.games.every(game => game.plies === 1 && game.reason === 'checkmate'));
  assert.equal(report.summary.completion.gameCompletionRate, 1);
  assert.equal(report.summary.completion.pairCompletionRate, 1);
  assert.equal(report.summary.completion.eligiblePairRate, 1);
});

test('arena plays completed and incomplete time-limited legal actions with both color assignments', async () => {
  for (const completed of [true, false]) {
    const engine = position => ({ ...firstLegal(position), completed, stoppedReason: 'time',
      status: completed ? 'ok' : 'incomplete', score: completed ? 42 : null });
    const report = await evaluateCandidate({ candidate: engine, incumbent: engine,
      suite: { cases: [{ id: 'tiny', position: tiny() }] }, pairs: 1, minPairs: 1, ...limits, maxPlies: 2 });
    assert.equal(report.limits.playOnTimeLimit, true);
    assert.deepEqual(report.games.map(game => game.aColor), [0, 1]);
    assert.deepEqual(report.games.map(game => game.moves.map(move => move.engine)), [['A', 'B'], ['B', 'A']]);
    assert.equal(report.summary.incompleteSearchMoves, completed ? 0 : 4);
    for (const game of report.games) {
      assert.equal(game.valid, true);
      assert.equal(game.plies, 2);
      assert.equal(game.reason, 'ply-limit');
      let replay = game.initialPosition;
      for (const move of game.moves) {
        assert.equal(move.search.stoppedReason, 'time');
        assert.equal(move.search.completed, completed);
        assert.equal(move.search.score, completed ? 42 : null);
        assert.equal(move.beforeKey, positionKey(replay));
        replay = validateAction(replay, move.action);
        assert.equal(move.afterKey, positionKey(replay));
      }
      assert.equal(game.finalKey, positionKey(replay));
    }
  }
});

test('arena time fallback does not forgive absent actions, illegal actions or invalid PVs', async () => {
  const illegal = [[[0, 0, 0, 0], [0, 0, 3, 3]]];
  for (const [patch, reason, valid] of [
    [{ bestAction: null, pv: [] }, 'time-limit', true],
    [{ bestAction: illegal, pv: [illegal] }, 'illegal-action', false],
    [{ pv: [illegal] }, 'invalid-pv', false],
  ]) {
    const engine = position => ({ ...firstLegal(position), completed: false,
      status: 'incomplete', stoppedReason: 'time', ...patch });
    const report = await evaluateCandidate({ candidate: engine, incumbent: engine,
      suite: { cases: [{ id: 'tiny', position: tiny() }] }, pairs: 1, minPairs: 1, ...limits });
    for (const game of report.games) {
      assert.equal(game.plies, 0);
      assert.equal(game.result, 'UNFINISHED');
      assert.equal(game.reason, reason);
      assert.equal(game.valid, valid);
    }
    assert.equal(report.decision.promote, false);
    assert.equal(report.decision.eligiblePairs, 0);
    assert.equal(report.decision.invalidGames, valid ? 0 : 2);
  }
});

test('real Transformer search retains and plays its legal fallback when inference reaches the deadline', async t => {
  let now = 0, inferenceCalls = 0;
  t.mock.method(performance, 'now', () => now);
  const engine = (position, options) => analyzeTransformer(position, { ...options, candidateLimit: 1,
    evaluateBatch: async positions => {
      inferenceCalls++;
      now += options.timeMs;
      return positions.map(() => 0);
    },
  });
  const report = await evaluateCandidate({ candidate: engine, incumbent: engine,
    suite: { cases: [{ id: 'tiny', position: tiny() }] }, pairs: 1, minPairs: 1, ...limits });
  assert.equal(inferenceCalls, 2);
  assert.equal(report.summary.incompleteSearchMoves, 2);
  for (const game of report.games) {
    assert.equal(game.valid, true);
    assert.equal(game.plies, 1);
    assert.equal(game.reason, 'ply-limit');
    assert.equal(game.moves[0].search.stoppedReason, 'time');
    assert.equal(game.moves[0].search.completed, false);
    assert.equal(game.moves[0].search.score, null);
    assert.equal(positionKey(validateAction(game.initialPosition, game.moves[0].action)), game.finalKey);
  }
});

test('cancellation rejects before games and while awaiting an unresponsive engine', async () => {
  const suite = { cases: [{ id: 'tiny', position: tiny() }] };
  await assert.rejects(evaluateCandidate({ candidate: firstLegal, incumbent: firstLegal, suite,
    shouldStop: () => true, ...limits }), { name: 'AbortError' });
  let cancelled = false;
  const timer = setTimeout(() => { cancelled = true; }, 40);
  try {
    await assert.rejects(evaluateCandidate({ candidate: () => new Promise(() => {}), incumbent: firstLegal,
      suite, shouldStop: () => cancelled, ...limits }), { name: 'AbortError' });
  } finally { clearTimeout(timer); }
});

test('parallel cancellation stops dispatch and drains started cooperative engines', { timeout: 5000 }, async () => {
  const allStarted = deferred();
  let started = 0, active = 0, stopped = 0, writes = 0, stop = false;
  const engine = async (position, { shouldStop }) => {
    active++;
    if (++started === 3) allStarted.resolve();
    try {
      await allStarted.promise;
      while (!shouldStop()) await nextTurn();
      stopped++;
      const error = new Error('Stopped'); error.name = 'AbortError'; throw error;
    } finally { active--; }
  };
  const evaluation = evaluateCandidate({ candidate: engine, incumbent: engine, suite: parallelSuite(),
    pairs: 2, gameConcurrency: 3, shouldStop: () => stop, ...limits,
    onGame: () => { writes++; },
  });
  const rejected = assert.rejects(evaluation, { name: 'AbortError' });
  await allStarted.promise;
  stop = true;
  await rejected;
  assert.equal(started, 3);
  assert.equal(stopped, 3);
  assert.equal(active, 0);
  assert.equal(writes, 0);
  await nextTurn();
  assert.equal(started, 3);
  assert.equal(writes, 0);
});

test('parallel cancellation waits for an in-flight completion callback before rejecting', { timeout: 5000 }, async () => {
  const allStarted = deferred(), writing = deferred(), finishWrite = deferred();
  let started = 0, active = 0, writes = 0, stop = false, settled = false;
  const engine = async (position, { shouldStop }) => {
    const index = started++;
    active++;
    if (started === 3) allStarted.resolve();
    try {
      await allStarted.promise;
      if (index === 0) return firstLegal(position);
      while (!shouldStop()) await nextTurn();
      const error = new Error('Stopped'); error.name = 'AbortError'; throw error;
    } finally { active--; }
  };
  const evaluation = evaluateCandidate({ candidate: engine, incumbent: engine, suite: parallelSuite(),
    pairs: 2, gameConcurrency: 3, shouldStop: () => stop, ...limits,
    onGame: async () => {
      writing.resolve();
      await finishWrite.promise;
      writes++;
    },
  });
  evaluation.then(() => { settled = true; }, () => { settled = true; });
  const rejected = assert.rejects(evaluation, { name: 'AbortError' });
  await writing.promise;
  stop = true;
  await nextTurn();
  assert.equal(settled, false, 'The arena must drain the callback before settling.');
  finishWrite.resolve();
  await rejected;
  assert.equal(started, 3);
  assert.equal(active, 0);
  assert.equal(writes, 1);
  await nextTurn();
  assert.equal(started, 3);
  assert.equal(writes, 1);
});

test('a parallel completion callback failure cancels sibling engines and preserves the failure', { timeout: 5000 }, async () => {
  const allStarted = deferred(), failure = new Error('Arena report unavailable');
  let started = 0, active = 0, stopped = 0, writes = 0;
  const engine = async (position, { shouldStop }) => {
    const index = started++;
    active++;
    if (started === 3) allStarted.resolve();
    try {
      await allStarted.promise;
      if (index === 0) return firstLegal(position);
      while (!shouldStop()) await nextTurn();
      stopped++;
      const error = new Error('Stopped'); error.name = 'AbortError'; throw error;
    } finally { active--; }
  };
  const evaluation = evaluateCandidate({ candidate: engine, incumbent: engine, suite: parallelSuite(),
    pairs: 2, gameConcurrency: 3, ...limits,
    onGame: async () => { writes++; await nextTurn(); throw failure; },
  });
  await assert.rejects(evaluation, error => error === failure);
  assert.equal(started, 3);
  assert.equal(stopped, 2);
  assert.equal(active, 0);
  assert.equal(writes, 1);
  await nextTurn();
  assert.equal(started, 3);
  assert.equal(writes, 1);
});

test('engine cancellation and mutation are handled without forgiving invalid games', async () => {
  const suite = { cases: [{ id: 'tiny', position: tiny() }] };
  await assert.rejects(evaluateCandidate({ candidate: position => ({ ...firstLegal(position), stoppedReason: 'cancelled' }),
    incumbent: firstLegal, suite, ...limits }), { name: 'AbortError' });
  for (const gameConcurrency of [1, 2]) {
    const report = await evaluateCandidate({ candidate: position => { position.action += 2; return firstLegal(position); },
      incumbent: firstLegal, suite, pairs: 1, minPairs: 1, gameConcurrency, ...limits });
    assert.equal(report.decision.reason, 'invalid-games');
    assert.equal(report.games[0].reason, 'input-mutation');
  }
});

test('invalid arena configuration is rejected before play', async () => {
  const base = { candidate: firstLegal, incumbent: firstLegal, suite: { cases: [{ id: 'tiny', position: tiny() }] } };
  for (const patch of [{ pairs: 0 }, { minPairs: 0 }, { seed: -1 }, { promotionScore: 0.49 }, { timeMs: 0 }, { maxDepth: -1 }, { maxDepth: 65 },
    { terminalTimeMs: 0 }, { terminalTimeMs: NaN },
    ...[0, 9, 1.5, NaN, '2'].map(gameConcurrency => ({ gameConcurrency }))]) {
    await assert.rejects(evaluateCandidate({ ...base, ...patch }));
  }
});
