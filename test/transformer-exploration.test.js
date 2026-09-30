import test from 'node:test';
import assert from 'node:assert/strict';
import { sampleExploration, actionCoverage } from '../scripts/transformer-exploration.js';
import { createPosition, generateActions, parseMove, validateAction } from '../src/rules.js';

const rng = seed => () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
const limits = { maxNodes: 20000, timeMs: 10000 };

test('random component traversal reaches legal compound turns beyond the fixed first 32', () => {
  const position = createPosition({ variant: 'two_timelines' }), before = structuredClone(position);
  const iterator = generateActions(position, { pruneUnsafe: false, skipOptionalSpatial: false });
  const prefix = new Set();
  try { for (let index = 0; index < 32; index++) prefix.add(JSON.stringify(iterator.next().value.moves)); }
  finally { iterator.return(); }
  const samples = [3, 7, 19, 41].map(seed => sampleExploration(position, rng(seed), limits));
  assert(samples.some(sample => !prefix.has(JSON.stringify(sample.action))));
  for (const [index, sample] of samples.entries()) {
    validateAction(position, sample.action);
    assert.equal(sample.coverage.actions, sample.candidates);
    assert(sample.coverage.compound > 0);
    assert(sample.selectedCoverage.compound > 0);
    assert(sample.work <= limits.maxNodes);
    assert.deepEqual(sample, sampleExploration(position, rng([3, 7, 19, 41][index]), limits));
  }
  assert.deepEqual(position, before);
});

test('coverage distinguishes a legal temporal branch from ordinary spatial play', () => {
  const position = createPosition({ pgn: '1. Nf3 / Nf6 2. Nc3 / Nc6' });
  const temporal = [parseMove(position, [[0, 4, 2, 2], [0, 2, 4, 2]])];
  assert.deepEqual(actionCoverage(position, temporal, validateAction(position, temporal)),
    { actions: 1, temporal: 1, compound: 0, branching: 1 });
  const standard = createPosition(), iterator = generateActions(standard);
  const spatial = iterator.next().value; iterator.return();
  assert.deepEqual(actionCoverage(standard, spatial.moves, spatial.position),
    { actions: 1, temporal: 0, compound: 0, branching: 0 });
});

test('exploration is bounded and cancellation propagates rather than becoming a sample', () => {
  const position = createPosition();
  const result = sampleExploration(position, rng(1), { ...limits, maxNodes: 0 });
  assert.equal(result.action, null);
  assert.equal(result.work, 0);
  assert.equal(result.stoppedReason, 'exploration-work-limit');
  const cancellation = new Error('cancelled');
  assert.throws(() => sampleExploration(position, rng(1), { ...limits, check() { throw cancellation; } }), error => error === cancellation);
});
