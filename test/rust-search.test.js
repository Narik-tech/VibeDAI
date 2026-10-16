import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { analyze as analyzeNative } from '../src/rust-engine.js';
import { analyze as analyzeReference } from '../src/search.js';
import { createPosition, positionKey, validateAction } from '../src/rules.js';
import { assessTactic, loadTactics } from '../scripts/strength.js';
import { nativeExecutable, nativeTestOptions } from './rust-native-helper.js';

const promotions = [10, 9, 8, 7, 6, 5, 4, 3];
const captureBoard = () => [[12, 0, 0, 0], [8, 0, 9, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
const capturePosition = () => ({ board: [[captureBoard()]], action: 0, promotions });
const historyPosition = () => ({ board: [[captureBoard(), captureBoard(), captureBoard()]], action: 0, promotions });
const limits = { timeMs: 0, unlimitedTime: true, maxNodes: 500_000, maxDepth: 2, quiescenceDepth: 0, mateProbeWork: 0 };
function valid(position, result) {
  const assessment = assessTactic({ position, expected: {} }, result);
  assert.equal(assessment.valid, true, assessment.errors.join(' '));
  if (result.bestAction) validateAction(position, result.bestAction);
  assert.equal(result.nodes, result.searchNodes + result.generationNodes);
  assert(result.nodes <= result.limits.maxNodes);
}

const suite = await loadTactics();
for (const fixture of suite.cases) {
  test(`native tactical corpus: ${fixture.title}`, nativeTestOptions, async () => {
    const before = positionKey(fixture.position);
    const result = await analyzeNative(fixture.position, { ...fixture.limits, timeMs: 30_000 });
    const assessment = assessTactic(fixture, result, before);
    assert.equal(assessment.valid, true, assessment.errors.join(' '));
    assert.equal(assessment.solved, true, `${assessment.failures.join(' ')} ${JSON.stringify(result)}`);
    assert.equal(positionKey(fixture.position), before);
  });
}

test('native fixed-depth search preserves reference scores and best actions across horizons', nativeTestOptions, async () => {
  const fixtures = [
    ['standard', createPosition()],
    ['opening', createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' })],
    ['mini capture', capturePosition()],
    ['historical capture', historyPosition()],
  ];
  for (const [name, position] of fixtures) for (const quiescenceDepth of [0, 1]) {
    const options = { ...limits, quiescenceDepth };
    const reference = analyzeReference(position, options);
    const result = await analyzeNative(position, options);
    valid(position, result);
    assert.equal(result.depth, 2, name);
    assert.equal(result.completed, true, name);
    assert.equal(result.effectiveQuiescenceDepth, quiescenceDepth, name);
    assert.equal(result.score, reference.score, `${name}, qdepth ${quiescenceDepth}`);
    assert.deepEqual(result.bestAction, reference.bestAction, `${name}, qdepth ${quiescenceDepth}`);
  }
});

test('native transposition bounds, aspiration retries and cache limits preserve complete results', nativeTestOptions, async () => {
  for (const position of [createPosition(), capturePosition(), historyPosition()]) {
    const options = { ...limits, maxDepth: 3, quiescenceDepth: 1, maxNodes: 1_000_000 };
    const reference = await analyzeNative(position, { ...options, maxTableEntries: 0 });
    assert.equal(reference.depth, 3);
    assert.equal(reference.tableEntries, 0);
    assert.equal(reference.cacheMemoryBytes, 0);
    for (const config of [
      {},
      { heuristics: { aspirationWindow: 1 } },
      { heuristics: { aspirationWindow: 0 }, cacheMemoryMb: 0 },
      { cacheMemoryMb: 0.01, maxTableEntries: 3 },
    ]) {
      const result = await analyzeNative(position, { ...options, ...config });
      valid(position, result);
      assert.equal(result.depth, 3);
      assert.equal(result.score, reference.score);
      assert.deepEqual(result.bestAction, reference.bestAction);
      assert(result.tableEntries <= result.limits.maxTableEntries);
      assert(result.cacheMemoryBytes <= result.limits.cacheMemoryMb * 1024 * 1024);
      if (config.cacheMemoryMb === 0) assert.equal(result.tableEntries, 0);
    }
  }
});

test('native scores retain White perspective for Black to move', nativeTestOptions, async () => {
  const white = capturePosition();
  const reflected = white.board[0][0].toReversed().map(row => row.map(piece =>
    piece === 0 ? 0 : piece % 2 ? piece + 1 : piece - 1));
  const black = { board: [[null, reflected]], action: 1, promotions };
  const options = { ...limits, quiescenceDepth: 1 };
  const result = await analyzeNative(black, options);
  const reference = analyzeReference(black, options);
  valid(black, result);
  assert(result.score < -500);
  assert.equal(result.score, reference.score);
  assert.deepEqual(result.bestAction, reference.bestAction);
});

test('native deterministic work budgets retain legal fallbacks without inventing terminals', nativeTestOptions, async () => {
  const position = createPosition();
  for (const budget of [0, 3, 20, 150]) {
    const options = { ...limits, maxNodes: budget, maxDepth: 8, quiescenceDepth: 2 };
    const first = await analyzeNative(position, options);
    const second = await analyzeNative(position, options);
    valid(position, first);
    assert.equal(first.nodes, budget);
    assert.equal(first.stoppedReason, 'nodes');
    for (const key of ['bestAction', 'pv', 'score', 'depth', 'nodes', 'searchNodes', 'generationNodes', 'qnodes', 'effectiveQuiescenceDepth']) {
      assert.deepEqual(first[key], second[key], `${key}, budget ${budget}`);
    }
    if (budget <= 3) {
      assert.equal(first.completed, false);
      assert.equal(first.score, null);
      assert.equal(first.status, 'incomplete');
      assert.equal(first.bestAction === null, budget === 0);
    }
  }
  const mate = createPosition({ pgn: '1. e3 / f6 2. Qe2 / Nc6 3. Qh5' });
  const result = await analyzeNative(mate, { ...limits, maxNodes: 0 });
  assert.equal(result.status, 'incomplete');
  assert.equal(result.completed, false);
  assert.equal(result.score, null);
  assert.equal(result.bestAction, null);
});

test('native time limits return promptly without exceeding the work budget', nativeTestOptions, async () => {
  const position = createPosition({ variant: 'two_timelines' });
  const started = performance.now();
  const result = await analyzeNative(position, { timeMs: 5, maxNodes: 1_000_000_000, maxDepth: 64, quiescenceDepth: 8, mateProbeWork: 0 });
  valid(position, result);
  assert.equal(result.stoppedReason, 'time');
  assert(performance.now() - started < 5_000, 'native time limit must terminate without waiting for a complete iteration');
});

test('native cooperative cancellation preserves its last completed iteration', nativeTestOptions, async () => {
  const position = createPosition({ variant: 'two_timelines' });
  let stop = false, lastCompleted;
  const result = await analyzeNative(position, {
    timeMs: 0, maxNodes: 1_000_000_000, maxDepth: 64, quiescenceDepth: 2, mateProbeWork: 0,
    shouldStop: () => stop,
    onProgress(progress) {
      if (progress.completed) { lastCompleted = progress; stop = true; }
    },
  });
  valid(position, result);
  assert.equal(result.stoppedReason, 'cancelled');
  assert.equal(result.completed, true);
  assert(lastCompleted, 'cancellation follows a completed native iteration');
  for (const key of ['bestAction', 'score', 'depth', 'effectiveQuiescenceDepth']) {
    assert.deepEqual(result[key], lastCompleted[key], key);
  }
});

test('native process stops an unlimited search when its input pipe closes', nativeTestOptions, () => {
  const child = spawnSync(nativeExecutable, [], {
    input: JSON.stringify({ type: 'analyze', position: createPosition({ variant: 'two_timelines' }),
      options: { timeMs: 0, maxNodes: 1_000_000_000, maxDepth: 64, quiescenceDepth: 8 } }) + '\n',
    encoding: 'utf8', timeout: 5_000, windowsHide: true,
  });
  assert.ifError(child.error);
  assert.equal(child.status, 0, child.stderr);
  const final = child.stdout.trim().split('\n').map(JSON.parse).find(message => message.type === 'result');
  assert(final);
  assert.equal(final.result.stoppedReason, 'cancelled');
});

test('native policy exhaustion remains incomplete while full-rules evasions exist', nativeTestOptions, async () => {
  const required = [[0, 11, 12], [7, 0, 0], [3, 0, 0]];
  const future = [[0, 8, 0], [11, 0, 0], [4, 0, 12]];
  const position = { board: [[required], null, [structuredClone(future), structuredClone(future), future]], action: 0, promotions };
  const save = [[[2, 2, 0, 1], [2, 2, 0, 2]], [[0, 0, 0, 2], [2, 2, 0, 2]]];
  validateAction(position, save);
  const result = await analyzeNative(position, limits);
  valid(position, result);
  assert.equal(result.status, 'incomplete');
  assert.equal(result.completed, false);
  assert.equal(result.stoppedReason, 'policy');
  assert.equal(result.score, null);
  assert.equal(result.bestAction, null);
  assert(result.policyLeaves > 0);
});
