import test from 'node:test';
import assert from 'node:assert/strict';
import { generateSelfPlayGames } from '../scripts/transformer-selfplay-games.js';
import { certifyTerminal } from '../scripts/match.js';
import { analyze } from '../src/transformer-search.js';
import { applyMove, createPosition, generateActions, positionKey, validateAction } from '../src/rules.js';

const limits = { games: 1, maxPlies: 2, timeMs: 3000, maxNodes: 20000, maxDepth: 1, exploration: 0 };
const starts = () => [{ id: 'standard', position: createPosition() }];
const neural = (position, options) => analyze(position, { ...options, evaluateBatch: async positions => positions.map(() => 0) });
function firstLegal(position, overrides = {}) {
  const iterator = generateActions(position);
  let action;
  try { action = iterator.next().value?.moves ?? null; }
  finally { iterator.return(); }
  return { engine: 'transformer', bestAction: action, pv: action ? [action] : [], completed: true,
    status: 'ok', score: 250, depth: 1, nodes: 2, searchNodes: 1, generationNodes: 1, stoppedReason: 'depth',
    searchPolicy: 'transformer-bounded-alpha-beta', ...overrides };
}
function mateStarts() {
  const white = createPosition({ pgn: '[Board "Custom"]\n[Size "4x4"]\n[Promotions "Q,R,B,N"]\n[3k/1P2/4/K3:0:1:w]' });
  const black = { board: [[null, white.board[0][0].toReversed().map(row => row.map(piece => piece === 0 ? 0 : piece % 2 ? piece + 1 : piece - 1))]],
    action: 1, promotions: white.promotions.slice() };
  return [{ id: 'white-mate', position: white }, { id: 'black-mate', position: black }];
}
function replay(game) {
  let position = game.initialPosition;
  for (const row of game.moves) {
    assert.equal(row.beforeKey, positionKey(position));
    position = validateAction(position, row.action);
    assert.equal(row.afterKey, positionKey(position));
  }
  assert.equal(positionKey(position), game.finalKey);
  return position;
}
const nextTurn = () => new Promise(resolve => setImmediate(resolve));
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

test('dynamic depth passes unchanged to self-play searches and saved game limits', async () => {
  const depths = [];
  const result = await generateSelfPlayGames({ ...limits, positions: starts(), maxDepth: 0,
    analyzePosition: (position, options) => {
      depths.push(options.maxDepth);
      return firstLegal(position, { limits: { maxDepth: options.maxDepth } });
    },
  });
  assert.deepEqual(depths, [0, 0]);
  assert.equal(result.games[0].limits.maxDepth, 0);
  assert(result.games[0].moves.every(move => move.search.limits.maxDepth === 0));
  assert.equal(result.games[0].valid, true);
});

test('self-play requests evaluated policy alternatives and keeps terminal time independent', async () => {
  const seen = [];
  const result = await generateSelfPlayGames({ ...limits, positions: starts(), maxPlies: 1, terminalTimeMs: 9000,
    analyzePosition(position, options) { seen.push(options); return firstLegal(position); },
  });
  assert.equal(seen.length, 1);
  assert.equal(seen[0].collectPolicyAlternatives, true);
  assert.equal(seen[0].timeMs, limits.timeMs);
  assert.equal(result.games[0].limits.terminalTimeMs, 9000);
  assert.deepEqual(result.summary.coverage, { actions: 1, temporal: 0, compound: 0, branching: 0 });
});

test('concurrent games bound overlapping searches, stream completion order and return stable indices', async () => {
  const firstGame = deferred(), seen = [];
  let launched = 0, active = 0, peak = 0, writing = 0, peakWriting = 0;
  const result = await generateSelfPlayGames({ ...limits, positions: starts(), games: 5, gameConcurrency: 2, maxPlies: 1,
    analyzePosition: async position => {
      const index = launched++;
      peak = Math.max(peak, ++active);
      try {
        if (index === 0) await firstGame.promise;
        return firstLegal(position);
      } finally { active--; }
    },
    onGame: async game => {
      peakWriting = Math.max(peakWriting, ++writing);
      await nextTurn();
      seen.push(game.index);
      if (game.index === 3) firstGame.resolve();
      writing--;
    },
  });
  assert.equal(peak, 2);
  assert.equal(active, 0);
  assert.equal(peakWriting, 1, 'Streaming writes must not overlap.');
  assert.deepEqual(seen.slice(0, 3), [1, 2, 3]);
  assert.deepEqual(result.games.map(game => game.index), [0, 1, 2, 3, 4]);
  assert.deepEqual(result.samples.map(sample => sample.gameId), result.games.map(game => game.gameId));
  assert.equal(result.summary.games, 5);
});

test('per-game exploration, start rotation and samples are identical across concurrency settings', async () => {
  const positions = [{ id: 'a', position: createPosition() }, { id: 'b', position: createPosition() }];
  const options = { ...limits, positions, games: 4, maxPlies: 3, exploration: 1, explorationPlies: 3, seed: 17,
    metadata: { runId: 'deterministic' },
    analyzePosition: async position => { await nextTurn(); return firstLegal(position); },
  };
  const serial = await generateSelfPlayGames({ ...options, gameConcurrency: 1 });
  const parallel = await generateSelfPlayGames({ ...options, gameConcurrency: 3 });
  const normalize = result => ({ ...result, games: result.games.map(game => ({ ...game,
    limits: { ...game.limits, gameConcurrency: 1 },
  })) });
  assert.deepEqual(normalize(parallel), normalize(serial));
  assert.deepEqual(parallel.games.map(game => game.startId), ['b', 'a', 'b', 'a']);
  assert.deepEqual(parallel.games.map(game => game.gameId), ['deterministic:17:1', 'deterministic:17:2', 'deterministic:17:3', 'deterministic:17:4']);
  for (const game of parallel.games) replay(game);
});

test('parallel cancellation stops assigning games and drains started lanes and streaming callbacks', async () => {
  const allStarted = deferred();
  let started = 0, active = 0, stoppedCallbacks = 0, writes = 0, stop = false;
  const generation = generateSelfPlayGames({ ...limits, positions: starts(), games: 10, gameConcurrency: 3,
    shouldStop: () => stop,
    analyzePosition: async (position, { shouldStop }) => {
      active++;
      if (++started === 3) allStarted.resolve();
      try {
        await allStarted.promise;
        while (!shouldStop()) await nextTurn();
        stoppedCallbacks++;
        const error = new Error('Stopped'); error.name = 'AbortError'; throw error;
      } finally { active--; }
    },
    onGame: async () => { await nextTurn(); writes++; },
  });
  await allStarted.promise;
  stop = true;
  const result = await generation;
  assert.equal(started, 3);
  assert.equal(stoppedCallbacks, 3);
  assert.equal(active, 0);
  assert.equal(writes, 3);
  assert.equal(result.summary.cancelled, true);
  assert.deepEqual(result.games.map(game => game.index), [0, 1, 2]);
  assert(result.games.every(game => game.reason === 'cancelled' && game.valid));
  await nextTurn();
  assert.equal(started, 3);
  assert.equal(writes, 3);
});

test('a streaming failure cancels parallel searches and drains lanes before rejecting', async () => {
  const allStarted = deferred(), failure = new Error('Output unavailable');
  let started = 0, active = 0, cancelled = 0, writes = 0;
  const generation = generateSelfPlayGames({ ...limits, positions: starts(), games: 10, gameConcurrency: 3, maxPlies: 1,
    analyzePosition: async (position, { shouldStop }) => {
      const index = started++;
      active++;
      if (started === 3) allStarted.resolve();
      try {
        await allStarted.promise;
        if (index === 0) return firstLegal(position);
        while (!shouldStop()) await nextTurn();
        cancelled++;
        const error = new Error('Stopped'); error.name = 'AbortError'; throw error;
      } finally { active--; }
    },
    onGame: async () => { writes++; await nextTurn(); throw failure; },
  });
  await assert.rejects(generation, error => error === failure);
  assert.equal(started, 3);
  assert.equal(cancelled, 2);
  assert.equal(active, 0);
  assert.equal(writes, 1);
  await nextTurn();
  assert.equal(started, 3);
  assert.equal(writes, 1);
});

test('actual Transformer self-play certifies both White and Black wins and blends White-relative targets', async () => {
  const positions = mateStarts(), before = structuredClone(positions), seen = [];
  const result = await generateSelfPlayGames({ ...limits, positions, games: 2, seed: 4, maxPlies: 1, outcomeWeight: 0.5,
    metadata: { runId: 'test-round', checkpointSha256: 'abc', trainingRound: 1 },
    analyzePosition: async (position, options) => ({ ...await neural(position, options), score: position.action % 2 ? -250 : 250 }),
    onGame: async (game, samples) => { await Promise.resolve(); seen.push({ game, samples }); },
  });
  assert.deepEqual(positions, before);
  assert.equal(result.summary.whiteWins, 1);
  assert.equal(result.summary.blackWins, 1);
  assert.equal(result.summary.outcomeSamples, 2);
  assert.equal(seen.length, 2);
  for (const [index, game] of result.games.entries()) {
    assert.equal(game.result, index ? 'BLACK_WIN' : 'WHITE_WIN');
    assert.equal(game.certificate.verified, true);
    assert.equal(certifyTerminal(replay(game)).status, 'checkmate');
    const sample = result.samples[index], outcome = index ? -1 : 1;
    assert.equal(sample.outcomeWhite, outcome);
    assert.equal(sample.value, 1000 * Math.atanh(0.5 * Math.tanh(outcome * 0.25) + 0.5 * outcome));
    assert.equal(sample.targetType, 'outcome-blend');
    assert.equal(sample.provenance.checkpointSha256, 'abc');
    assert.equal(sample.gameId, `test-round:4:${index + 1}`);
    assert.equal(sample.searchScoreWhiteCp, outcome * 250);
    assert.deepEqual(seen[index].samples, [sample]);
  }
});

test('certified stalemate blends toward zero instead of treating a cap as a draw', async () => {
  const position = createPosition({ pgn: '[Board "Custom"]\n[Size "4x4"]\n[3k/4/1Q2/K3:0:1:w]' });
  const action = [[[0, 0, 1, 1], [0, 0, 1, 2]]];
  const result = await generateSelfPlayGames({ ...limits, positions: [{ id: 'draw', position }], maxPlies: 1,
    analyzePosition: pos => firstLegal(pos, { bestAction: action, pv: [action], score: 400 }),
  });
  assert.equal(result.games[0].result, 'DRAW');
  assert.equal(result.games[0].certificate.status, 'stalemate');
  assert.equal(result.samples[0].outcomeWhite, 0);
  assert.equal(result.samples[0].value, 1000 * Math.atanh(0.5 * Math.tanh(0.4)));
});

test('unfinished games bootstrap completed root search values without outcomes', async () => {
  const positions = starts(), before = structuredClone(positions);
  const result = await generateSelfPlayGames({ ...limits, positions, analyzePosition: position => firstLegal(position), outcomeWeight: 1 });
  assert.deepEqual(positions, before);
  assert.equal(result.games[0].result, 'UNFINISHED');
  assert.equal(result.games[0].reason, 'ply-limit');
  assert.equal(result.summary.draws, 0);
  assert.equal(result.samples.length, 2);
  for (const sample of result.samples) {
    assert.equal(sample.targetType, 'search-bootstrap');
    assert.equal(sample.value, 250);
    assert.equal(sample.outcomeWhite, null);
    assert.equal(sample.outcomeWeight, 0);
    assert.equal(sample.gameResult, 'UNFINISHED');
  }
  assert.equal(result.samples[1].position.action % 2, 1);
  assert.equal(result.samples[1].value, 250, 'Black-to-move does not flip a White-relative score.');
  replay(result.games[0]);
});

test('unverified terminal games never receive outcomes even after an apparent mate', async () => {
  const result = await generateSelfPlayGames({ ...limits, positions: mateStarts().slice(0, 1), maxPlies: 1, terminalWork: 0, analyzePosition: neural });
  assert.equal(result.games[0].result, 'UNFINISHED');
  assert.equal(result.games[0].reason, 'terminal-work-limit');
  assert.equal(result.games[0].certificate.verified, false);
  assert.equal(result.samples[0].targetType, 'search-bootstrap');
  assert.equal(result.samples[0].value, 99999);
  assert.equal(result.samples[0].outcomeWhite, null);
});

test('seeded epsilon exploration is reproducible and preserves the searched-root target', async () => {
  const options = { ...limits, positions: starts(), analyzePosition: position => firstLegal(position), games: 2, maxPlies: 4,
    exploration: 1, explorationPlies: 4, seed: 17 };
  const first = await generateSelfPlayGames(options), second = await generateSelfPlayGames(options);
  const trace = output => output.games.map(game => game.moves.map(row => row.action));
  assert.deepEqual(trace(first), trace(second));
  assert.deepEqual(first.samples, second.samples);
  assert(first.games.flatMap(game => game.moves).every(row => row.exploration.explored));
  assert(first.samples.some(row => !assertSame(row.playedAction, row.searchedAction)));
  assert(first.samples.every(row => row.value === 250 && row.searchScoreWhiteCp === 250));
  for (const row of first.samples) {
    assert.equal(row.policyVersion, 1);
    assert(row.policy.length > 0);
    let prefix = row.position;
    for (const [index, action] of row.searchedAction.entries()) {
      const label = row.policy.find(item => item.componentIndex === index);
      if (label) {
        assert.deepEqual(label.position ?? row.position, prefix);
        assert.deepEqual(label.moves[label.target], action);
      }
      prefix = applyMove(prefix, action);
    }
  }
  for (const game of first.games) replay(game);
  function assertSame(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
});

test('compound self-play policy targets retain searched prefix order across boards', async () => {
  const position = createPosition({variant:'two_timelines'});
  const result = await generateSelfPlayGames({...limits, positions:[{id:'compound', position}], maxPlies:1,
    analyzePosition: position => firstLegal(position)});
  assert.equal(result.samples.length, 1);
  const row = result.samples[0];
  assert.equal(row.policy.length, row.searchedAction.length);
  assert(row.policy.length > 1);
  let prefix = row.position;
  for (const [index, label] of row.policy.entries()) {
    assert.deepEqual(label.position ?? row.position, prefix);
    assert.deepEqual(label.moves[label.target], row.searchedAction[index]);
    prefix = applyMove(prefix, row.searchedAction[index]);
  }
});

test('incomplete search scores are never used as training targets', async () => {
  const result = await generateSelfPlayGames({ ...limits, positions: starts(), analyzePosition: position => firstLegal(position, { completed: false, score: 777, status: 'incomplete', stoppedReason: 'nodes' }) });
  assert.equal(result.games[0].plies, 2);
  assert.equal(result.games[0].valid, true);
  assert.equal(result.samples.length, 0);
});

test('invalid later actions, PVs, mutation, work counters, scores and model errors discard every game sample', async () => {
  for (const [label, bad] of [
    ['model-error', () => { throw new Error('Model failed'); }],
    ['illegal-action', position => firstLegal(position, { bestAction: [[[99, 0, 0, 0], [99, 0, 1, 0]]] })],
    ['invalid-pv', position => { const result = firstLegal(position); result.pv.push([[[99, 0, 0, 0], [99, 0, 1, 0]]]); return result; }],
    ['input-mutation', position => { position.promotions.reverse(); return firstLegal(position); }],
    ['invalid-search-result', position => firstLegal(position, { engine: 'classical' })],
    ['invalid-work-accounting', position => firstLegal(position, { nodes: 999999 })],
    ['invalid-search-score', position => firstLegal(position, { score: NaN })],
    ['false-terminal-claim', position => firstLegal(position, { status: 'checkmate', bestAction: null, pv: [] })],
  ]) {
    let calls = 0, streamed;
    const positions = starts(), original = structuredClone(positions);
    const result = await generateSelfPlayGames({ ...limits, positions,
      analyzePosition: position => ++calls === 1 ? firstLegal(position) : bad(position),
      onGame: (game, samples) => { streamed = { game, samples }; },
    });
    assert.deepEqual(positions, original, label);
    assert.equal(result.games[0].reason, label);
    assert.equal(result.games[0].valid, false, label);
    assert.equal(result.games[0].result, 'UNFINISHED', label);
    assert.equal(result.games[0].discardedSamples, 1, label);
    assert.equal(result.samples.length, 0, label);
    assert.deepEqual(streamed.samples, [], label);
  }
});

test('cancellation keeps earlier verified search bootstraps but never game outcomes', async () => {
  let calls = 0, stop = false;
  const result = await generateSelfPlayGames({ ...limits, positions: starts(), games: 3, shouldStop: () => stop,
    analyzePosition: position => { if (++calls === 2) stop = true; return firstLegal(position); },
  });
  assert.equal(result.summary.cancelled, true);
  assert.equal(result.games.length, 1);
  assert.equal(result.games[0].reason, 'cancelled');
  assert.equal(result.games[0].result, 'UNFINISHED');
  assert.equal(result.samples.length, 1);
  assert.equal(result.samples[0].targetType, 'search-bootstrap');
  assert.equal(result.samples[0].outcomeWhite, null);
});

test('a cancelled worker rejection is cancellation instead of an invalid-model game', async () => {
  let calls = 0, stop = false;
  const result = await generateSelfPlayGames({ ...limits, positions: starts(), shouldStop: () => stop,
    analyzePosition: position => {
      if (++calls === 1) return firstLegal(position);
      stop = true;
      const error = new Error('Worker stopped'); error.name = 'AbortError'; throw error;
    },
  });
  assert.equal(result.games[0].reason, 'cancelled');
  assert.equal(result.games[0].valid, true);
  assert.equal(result.samples.length, 1);
  assert.equal(result.samples[0].targetType, 'search-bootstrap');
});

test('a cancelled terminal search cannot finish the game', async () => {
  const position = createPosition({ pgn: '[Board "Custom"]\n[Size "3x3"]\n[1q1/2k/K2:0:1:w]' });
  const result = await generateSelfPlayGames({ ...limits, positions: [{ id: 'terminal', position }],
    analyzePosition: () => ({ engine: 'transformer', status: 'stalemate', completed: true,
      score: 0, bestAction: null, pv: [], stoppedReason: 'cancelled' }),
  });
  assert.equal(result.games[0].result, 'UNFINISHED');
  assert.equal(result.games[0].reason, 'cancelled');
  assert.equal(result.summary.draws, 0);
});

test('cancellation during a stalled model callback returns promptly', async () => {
  let stop = false;
  const timer = setTimeout(() => { stop = true; }, 25), started = performance.now();
  try {
    const result = await generateSelfPlayGames({ ...limits, positions: starts(), shouldStop: () => stop,
      analyzePosition: () => new Promise(() => {}),
    });
    assert.equal(result.games[0].reason, 'cancelled');
    assert.equal(result.summary.cancelled, true);
    assert.equal(result.samples.length, 0);
    assert(performance.now() - started < 1000);
  } finally { clearTimeout(timer); }
});

test('exploration observes cancellation during unrestricted generation ticks', async () => {
  let polls = 0;
  const result = await generateSelfPlayGames({ ...limits, positions: starts(), exploration: 1,
    shouldStop: () => ++polls >= 25, analyzePosition: position => firstLegal(position),
  });
  assert.equal(result.games[0].reason, 'cancelled');
  assert.equal(result.games[0].plies, 0);
  assert.equal(result.samples.length, 0);
});

test('an already-cancelled run launches no games and bounded model stalls remain unfinished', async () => {
  const cancelled = await generateSelfPlayGames({ ...limits, positions: starts(), shouldStop: () => true,
    analyzePosition: () => { throw new Error('Must not be called'); },
  });
  assert.equal(cancelled.games.length, 0);
  assert.equal(cancelled.summary.cancelled, true);
  const stalled = await generateSelfPlayGames({ ...limits, positions: starts(), timeMs: 10,
    analyzePosition: () => new Promise(() => {}),
  });
  assert.equal(stalled.games[0].result, 'UNFINISHED');
  assert.equal(stalled.games[0].reason, 'search-time-limit');
  assert.equal(stalled.games[0].valid, true);
});

test('multi-board self-play keeps complete legal submissions and finite extreme-score targets', async () => {
  const position = createPosition({ variant: 'two_timelines' });
  const result = await generateSelfPlayGames({ ...limits, positions: [{ id: 'multi', position }], maxPlies: 1, analyzePosition: position => firstLegal(position) });
  assert(result.games[0].moves[0].action.length >= 2);
  replay(result.games[0]);
  const mate = await generateSelfPlayGames({ ...limits, positions: mateStarts().slice(0, 1), maxPlies: 1, outcomeWeight: 1, analyzePosition: neural });
  assert(Number.isFinite(mate.samples[0].value));
  assert.equal(mate.samples[0].value, 1000 * Math.atanh(0.999));
});

test('stream callback is awaited and invalid options are rejected', async () => {
  let streamed = false;
  await generateSelfPlayGames({ ...limits, positions: starts(), analyzePosition: position => firstLegal(position), maxPlies: 0,
    onGame: async () => { await new Promise(resolve => setTimeout(resolve, 5)); streamed = true; },
  });
  assert.equal(streamed, true);
  for (const bad of [{ games: 0 }, { gameConcurrency: 0 }, { gameConcurrency: 9 }, { gameConcurrency: 1.5 },
    { exploration: 2 }, { outcomeWeight: NaN }, { maxNodes: -1 }, { maxDepth: -1 }, { maxDepth: 65 }, { seed: -2 }, { timeMs: 0 }]) {
    await assert.rejects(generateSelfPlayGames({ ...limits, positions: starts(), analyzePosition: firstLegal, ...bad }), /Invalid/);
  }
  await assert.rejects(generateSelfPlayGames({ ...limits, positions: [], analyzePosition: firstLegal }), /positions/);
});
