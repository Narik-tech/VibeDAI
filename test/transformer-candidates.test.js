import test from 'node:test';
import assert from 'node:assert/strict';
import { createCandidateStream } from '../src/transformer-candidates.js';
import {
  applyMove, createPosition, generateActions, generateActionsAsync,
  positionKey, pseudoMoves, raw, validateAction,
} from '../src/rules.js';

const square = () => [[12, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
const position = board => ({ board, action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3] });
const keys = candidates => candidates.map(candidate => positionKey(candidate.position)).sort();
function harness(start, options = {}) {
  const cache = options.cache ?? new Map(), batches = [];
  const stream = createCandidateStream(start, {
    keyPosition: positionKey, valueFor: state => cache.get(positionKey(state)),
    async infer(states) {
      batches.push(states);
      for (const state of states) cache.set(positionKey(state), options.evaluate?.(state) ?? 0);
    },
    ...options,
  });
  return { stream, batches, cache };
}
function optionalKinds(start, moves) {
  const kinds = [];
  for (const move of moves) {
    if (!raw.boardFuncs.present(start.board, start.action).includes(move[0][0])) {
      kinds.push(move[0][0] === move[1][0] && move[0][1] === move[1][1] ? 'spatial' : 'temporal');
    }
    start = applyMove(start, move);
  }
  return kinds;
}

test('candidate stream yields before scoring every component and resumes without duplicates', async () => {
  const start = createPosition(), { stream, batches } = harness(start, { componentBatchSize: 3 });
  const first = await stream.next();
  assert.equal(first.done, false);
  assert.equal(batches.flat().length, 3);
  assert(batches.flat().length < pseudoMoves(start).length);
  const actual = [first.value];
  for await (const candidate of stream) actual.push(candidate);
  assert.deepEqual(keys(actual), keys([...generateActions(start)]));
  assert.equal(new Set(keys(actual)).size, actual.length);
  assert(batches.every(batch => batch.length <= 3));
  assert.equal(stream.selective, false);
});

test('optional temporal candidates get a reserved slot and optional spatial moves are omitted', async () => {
  const start = position([[square()], null, [square(), square(), square()]]);
  const { stream } = harness(start, { componentBatchSize: 2 });
  const actual = [];
  for await (const candidate of stream) {
    actual.push(candidate);
    assert(!optionalKinds(start, candidate.moves).includes('spatial'));
    assert.equal(positionKey(validateAction(start, candidate.moves)), positionKey(candidate.position));
  }
  assert(actual.slice(0, 4).some(candidate => optionalKinds(start, candidate.moves).includes('temporal')));
  assert(actual.slice(0, 3).every(candidate => !optionalKinds(start, candidate.moves).length));
  assert.deepEqual(keys(actual), keys([...generateActions(start, { skipOptionalSpatial: true })]));
  assert.equal(new Set(keys(actual)).size, actual.length);
  assert.equal(stream.selective, true);
});

test('completed candidates use submitted-state values rather than prefix averages', async () => {
  const start = position([[square()], null, [square()]]);
  const { stream, batches } = harness(start, { evaluate: state => state.action === 0 ? 1000 : -123 });
  let compound = false;
  for await (const candidate of stream) {
    compound ||= candidate.moves.length > 1;
    assert.equal(candidate.candidateScore, -123);
  }
  assert(compound);
  assert(batches.flat().some(state => state.action === 0));
});

test('history-cached values are reused and the trained policy selects the first small batch', async () => {
  const start = createPosition(), moves = pseudoMoves(start), favorite = JSON.stringify(moves.at(-1));
  const cache = new Map([...generateActions(start)].map(candidate => [positionKey(candidate.position), 12]));
  let policyCalls = 0;
  const { stream, batches } = harness(start, { cache, componentBatchSize: 2,
    async scoreMoves(_current, candidates) {
      policyCalls++;
      return candidates.map(move => JSON.stringify(move) === favorite ? 100 : 0);
    },
  });
  const first = await stream.next();
  assert.equal(JSON.stringify(first.value.moves[0]), favorite);
  assert.equal(first.value.candidateScore, 12);
  assert.equal(policyCalls, 1);
  assert.equal(batches.length, 0);
  await stream.return();
});

test('a bounded first batch includes later policy ranks without extra inference', async () => {
  const start = createPosition(), moves = pseudoMoves(start);
  const last = moves.at(-1), favorite = positionKey(validateAction(start, [last]));
  const { stream, batches } = harness(start, { componentBatchSize: 8,
    scoreMoves: async (_state, candidates) => candidates.map((_move, index) => -index),
    evaluate: state => positionKey(state) === favorite ? 500 : 0,
  });
  const first = await stream.next();
  assert.deepEqual(first.value.moves, [last], 'a misranked strong tail move must reach the admitted candidates');
  assert.equal(batches.flat().length, 8, 'diversity does not enlarge the inference batch');
  await stream.return();
});

test('multi-board admission alternates first-move branches and retains exhaustive legal coverage', async () => {
  const start = position([[square()], null, [square()]]);
  for (const branches of [1, 4]) {
    let rootPolicyCalls = 0;
    const { stream, batches } = harness(start, { componentBatchSize: 4, candidateBranches: branches,
      scoreMoves: async (_state, moves) => {
        if (_state === start) rootPolicyCalls++;
        return moves.map(() => 0);
      },
    });
    const actual = [];
    for await (const candidate of stream) actual.push(candidate);
    assert.deepEqual(keys(actual), keys([...generateActions(start, { skipOptionalSpatial: true })]));
    assert.equal(new Set(keys(actual)).size, actual.length);
    if (branches === 4) assert(new Set(actual.slice(0, 4).map(item => JSON.stringify(item.moves[0]))).size > 1);
    assert.equal(rootPolicyCalls, 1, 'lanes share the root policy and component batches');
    const evaluated = batches.flat().map(positionKey);
    assert.equal(new Set(evaluated).size, evaluated.length, 'lanes reuse exact-history neural values');
  }
});

test('async move batches preserve shared legal traversal and close nested generators on cancellation', async () => {
  const start = position([[square()], null, [square()]]);
  let opened = 0, closed = 0;
  const options = { orderMoves: async function* (_current, moves) {
    opened++;
    try { for (const move of moves) yield [move]; }
    finally { closed++; }
  } };
  const actual = [];
  for await (const candidate of generateActionsAsync(start, options)) actual.push(candidate);
  assert.deepEqual(keys(actual), keys([...generateActions(start)]));
  assert.equal(opened, closed);
  opened = 0; closed = 0;
  const paused = generateActionsAsync(start, options);
  assert.equal((await paused.next()).done, false);
  assert(opened > 0 && opened > closed);
  await paused.return();
  assert.equal(opened, closed);
});

test('selective exhaustion cannot turn an omitted spatial evasion into terminal proof', async () => {
  const required = [[0, 11, 12], [7, 0, 0], [3, 0, 0]];
  const future = [[0, 8, 0], [11, 0, 0], [4, 0, 12]];
  const start = position([[required], null, [structuredClone(future), structuredClone(future), future]]);
  const { stream } = harness(start);
  assert([...generateActions(start)].length > 0);
  assert.equal((await stream.next()).done, true);
  assert.equal(stream.selective, true);
});
