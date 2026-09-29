import test from 'node:test';
import assert from 'node:assert/strict';
import { analyze, createSearchSession } from '../src/search.js';
import { analyze as analyzeParallel } from '../src/parallel-search.js';
import { DEFAULT_HEURISTICS, normalizeHeuristics } from '../src/heuristics.js';
import { createPosition, formatAction, validateAction } from '../src/rules.js';

const limits = { unlimitedTime: true, maxNodes: 500000, maxDepth: 2 };
const capturePosition = () => ({
  board: [[[[12, 0, 0, 0], [8, 0, 9, 0], [0, 0, 0, 0], [0, 0, 0, 11]]]],
  action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3],
});

function validatePv(position, result) {
  for (const action of result.pv) position = validateAction(position, action);
}

test('default heuristic settings preserve the opening result and search work', () => {
  const position = createPosition();
  const options = { ...limits, quiescenceDepth: 0 };
  const implicit = analyze(position, options);
  const explicit = analyze(position, { ...options, heuristics: DEFAULT_HEURISTICS });
  assert.equal(implicit.depth, 2);
  assert.equal(implicit.score, 0);
  assert.equal(formatAction(position, implicit.bestAction), '(0T1)Nf3');
  for (const key of ['score', 'nodes', 'depth', 'effectiveQuiescenceDepth']) assert.equal(explicit[key], implicit[key]);
  assert.deepEqual(explicit.pv, implicit.pv);
  assert.deepEqual(explicit.heuristics, { ...DEFAULT_HEURISTICS, quiescenceDepth: 0 });
  validatePv(position, explicit);
});

test('tuned material values change search evaluation and remain isolated between analyses', () => {
  const position = capturePosition();
  const baseline = analyze(position, { ...limits, quiescenceDepth: 0 });
  const tuned = analyze(position, { ...limits, heuristics: { rookValue: 1100, quiescenceDepth: 0 } });
  const restored = analyze(position, { ...limits, quiescenceDepth: 0 });
  assert.equal(tuned.depth, 2);
  assert(tuned.score > baseline.score + 300, 'the remaining rook must use its customized material value');
  assert.deepEqual(tuned.bestAction, baseline.bestAction);
  assert.equal(restored.score, baseline.score);
  assert.deepEqual(restored.bestAction, baseline.bestAction);
  assert.equal(restored.nodes, baseline.nodes);
  assert.equal(tuned.heuristics.rookValue, 1100);
  validatePv(position, tuned);
});

test('quiet centralization tuning changes the legal fallback when the work budget is exhausted', () => {
  const position = createPosition({ pgn: '[Board "Custom"]\n[k7/8/8/8/8/5N2/8/K7:0:1:w]' });
  const options = { ...limits, maxNodes: 3 };
  const central = analyze(position, options);
  const neutral = analyze(position, { ...options, heuristics: { quietCentralization: 0 } });
  assert.equal(formatAction(position, central.bestAction), '(0T1)Kb2');
  assert.equal(formatAction(position, neutral.bestAction), '(0T1)Kb1');
  for (const result of [central, neutral]) {
    assert.equal(result.stoppedReason, 'nodes');
    assert.equal(result.score, null);
    validatePv(position, result);
  }
});

test('zero-valued captures retain tactical priority over quiet fallbacks', () => {
  const position = capturePosition();
  const result = analyze(position, { ...limits, maxNodes: 3, heuristics: { queenValue: 0 } });
  assert.deepEqual(result.bestAction, [[[0, 0, 1, 0], [0, 0, 1, 2]]]);
  validatePv(position, result);
});

test('heuristic quiescence depth is used unless the legacy option explicitly overrides it', () => {
  const position = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  const configured = analyze(position, { ...limits, maxDepth: 1, heuristics: { quiescenceDepth: 1 } });
  const legacy = analyze(position, { ...limits, maxDepth: 1, quiescenceDepth: 1 });
  const overridden = analyze(position, { ...limits, maxDepth: 1, quiescenceDepth: 0, heuristics: { quiescenceDepth: 1 } });
  assert.equal(configured.effectiveQuiescenceDepth, 1);
  assert.equal(configured.score, legacy.score);
  assert.deepEqual(configured.bestAction, legacy.bestAction);
  assert.equal(overridden.effectiveQuiescenceDepth, 0);
  assert.equal(overridden.limits.quiescenceDepth, 0);
  assert.equal(overridden.heuristics.quiescenceDepth, 0);
  assert.notEqual(overridden.score, configured.score, 'the fixture must expose the recapture horizon');
  const maximum = analyze(position, { ...limits, maxNodes: 0, quiescenceDepth: 8 });
  assert.equal(maximum.limits.quiescenceDepth, 8);
  assert.equal(maximum.heuristics.quiescenceDepth, 8);
});

test('aspiration zero searches full windows while configured widths surround the previous score', () => {
  const position = createPosition();
  function run(aspirationWindow) {
    const session = createSearchSession(position, { ...limits, heuristics: { quiescenceDepth: 0, aspirationWindow } });
    const iterations = session.iterations(), requests = [];
    let step = iterations.next();
    while (!step.done) {
      requests.push(step.value);
      step = iterations.next(session.root(step.value));
    }
    return { requests, result: step.value };
  }
  const full = run(0), narrow = run(25);
  assert.equal(full.requests[1].alpha, -1000000);
  assert.equal(full.requests[1].beta, 1000000);
  assert.equal(narrow.requests[1].beta - narrow.requests[1].alpha, 50);
  assert.equal(narrow.result.score, full.result.score);
  assert.deepEqual(narrow.result.bestAction, full.result.bestAction);
});

test('custom heuristics agree across serial, parallel, and uncached search and progress snapshots', async () => {
  const position = capturePosition(), original = structuredClone(position);
  const heuristics = normalizeHeuristics({
    rookValue: 1100, materialWeight: 1.5, quiescenceDepth: 1, aspirationWindow: 0,
    temporalMovePenalty: 0, quietCentralization: 30, killerBonus: 0, historyBonus: 0,
  });
  const options = { ...limits, heuristics };
  const reference = analyze(position, options);
  for (const cacheMemoryMb of [0, 128]) {
    const serial = analyze(position, { ...options, cacheMemoryMb });
    const snapshots = [];
    const parallel = await analyzeParallel(position, {
      ...options, cacheMemoryMb, threads: 2, onProgress: result => snapshots.push(result),
    });
    assert.equal(parallel.threadsUsed, 2, 'the test must dispatch work to a separate search session');
    for (const result of [serial, parallel]) {
      assert.equal(result.depth, 2);
      assert.equal(result.score, reference.score);
      assert.deepEqual(result.bestAction, reference.bestAction);
      assert.deepEqual(result.heuristics, heuristics);
      assert.equal(result.effectiveQuiescenceDepth, 1);
      if (!cacheMemoryMb) assert.equal(result.tableEntries, 0);
      validatePv(position, result);
    }
    for (const progress of snapshots) assert.deepEqual(progress.heuristics, heuristics);
  }
  assert.deepEqual(position, original);
});

test('invalid heuristic objects are rejected even when a legacy depth override is supplied', () => {
  const position = createPosition();
  for (const heuristics of [null, { rookValue: NaN }, { unknown: 1 }]) {
    assert.throws(() => analyze(position, { ...limits, maxNodes: 0, quiescenceDepth: 0, heuristics }));
  }
});
