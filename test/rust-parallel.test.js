import test from 'node:test';
import assert from 'node:assert/strict';
import { analyze } from '../src/rust-engine.js';
import { applyMove, createPosition, presentTimelines, submitPosition, validateAction } from '../src/rules.js';
import { nativeTestOptions } from './rust-native-helper.js';

const promotions = [10, 9, 8, 7, 6, 5, 4, 3];
const limits = { timeMs: 10_000, maxNodes: 500_000, maxDepth: 2, quiescenceDepth: 1 };
const capture = () => ({
  board: [[[[12, 0, 0, 0], [8, 0, 9, 0], [0, 0, 0, 0], [0, 0, 0, 11]]]],
  action: 0, promotions,
});

function validatePv(start, result) {
  let position = start;
  for (const action of result.pv) {
    validateAction(position, action);
    for (const move of action) {
      if (move[0][0] === move[1][0] && move[0][1] === move[1][1]) {
        assert.ok(presentTimelines(position).includes(move[0][0]), 'ordinary moves obey the present-board search policy');
      }
      position = applyMove(position, move);
    }
    position = submitPosition(position);
  }
}

function accounting(result, maxNodes, cacheMemoryMb = 128, maxTableEntries = 100_000) {
  assert.equal(result.nodes, result.searchNodes + result.generationNodes);
  assert.ok(result.nodes <= maxNodes, `global node budget: ${result.nodes} <= ${maxNodes}`);
  assert.ok(result.qnodes <= result.searchNodes);
  assert.ok(result.qTtHits <= result.ttHits);
  assert.ok(result.cacheMemoryBytes <= cacheMemoryMb * 1024 * 1024);
  assert.ok(result.tableEntries <= maxTableEntries);
}

test('native root workers match serial tactical scores and unique best moves for both colors', nativeTestOptions, async () => {
  const white = capture();
  const blackBoard = white.board[0][0].toReversed().map(row => row.map(piece => !piece ? 0 : piece % 2 ? piece + 1 : piece - 1));
  for (const position of [white, { board: [[null, blackBoard]], action: 1, promotions }]) {
    const before = structuredClone(position);
    const serial = await analyze(position, { ...limits, threads: 1 });
    assert.equal(serial.depth, 2);
    for (const threads of [2, 4]) {
      const result = await analyze(position, { ...limits, threads });
      assert.equal(result.depth, 2);
      assert.equal(result.completed, true);
      assert.equal(result.score, serial.score);
      assert.deepEqual(result.bestAction, serial.bestAction);
      assert.equal(result.limits.threads, threads);
      assert.equal(result.threadsUsed, threads);
      assert.deepEqual(position, before);
      validatePv(position, result);
      accounting(result, limits.maxNodes);
    }
  }
});

test('native parallel roots submit every required board and preserve serial scores', nativeTestOptions, async () => {
  const squares = capture().board[0][0];
  const position = { board: [[structuredClone(squares)], null, [structuredClone(squares)]], action: 0, promotions };
  const options = { ...limits, quiescenceDepth: 0 };
  const serial = await analyze(position, options);
  const result = await analyze(position, { ...options, threads: 2 });
  assert.equal(result.depth, 2);
  assert.equal(result.score, serial.score);
  assert.ok(result.bestAction.length >= 2);
  assert.equal(validateAction(position, result.bestAction).action, 1);
  validatePv(position, result);
  accounting(result, limits.maxNodes);
});

test('native threads share strict work, cache memory and entry budgets', nativeTestOptions, async () => {
  const position = createPosition();
  for (const maxNodes of [0, 1, 3, 1000, 5000]) {
    const result = await analyze(position, {
      ...limits, threads: 4, maxDepth: 64, maxNodes, cacheMemoryMb: 0.02, maxTableEntries: 50,
      onProgress: progress => accounting(progress, maxNodes, 0.02, 50),
    });
    assert.equal(result.nodes, maxNodes);
    assert.equal(result.stoppedReason, 'nodes');
    accounting(result, maxNodes, 0.02, 50);
    validatePv(position, result);
  }
  const uncached = await analyze(capture(), { ...limits, threads: 4, cacheMemoryMb: 0 });
  assert.equal(uncached.cacheMemoryBytes, 0);
  assert.equal(uncached.tableEntries, 0);
});

test('native parallel cancellation retains the last completed depth and quiescence horizon', nativeTestOptions, async () => {
  let stop = false, completed;
  const position = createPosition();
  const result = await analyze(position, {
    timeMs: 0, threads: 4, maxDepth: 64, maxNodes: 1_000_000_000, quiescenceDepth: 2,
    shouldStop: () => stop,
    onProgress: progress => {
      if (progress.completed) completed = progress;
      if (progress.depth >= 2) stop = true;
    },
  });
  assert.equal(result.stoppedReason, 'cancelled');
  assert.ok(completed);
  for (const field of ['depth', 'score', 'effectiveQuiescenceDepth']) assert.equal(result[field], completed[field], field);
  assert.deepEqual(result.bestAction, completed.bestAction);
  assert.deepEqual(result.pv, completed.pv);
  assert.equal(result.threadsUsed, 4);
  validatePv(position, result);
});

test('native thread settings are normalized to one through sixteen', nativeTestOptions, async () => {
  for (const [threads, expected] of [[0, 1], [-3, 1], [2.9, 2], [32, 16]]) {
    const result = await analyze(capture(), { threads, maxNodes: 0 });
    assert.equal(result.limits.threads, expected);
    assert.equal(result.nodes, 0);
  }
});
