import test from 'node:test';
import assert from 'node:assert/strict';
import { analyze } from '../src/search.js';
import { createPosition, validateAction } from '../src/rules.js';

const limits = { unlimitedTime: true, maxDepth: 2, maxNodes: 100000 };

function verify(position, result, nodes, pv) {
  assert.equal(result.depth, 2);
  assert.equal(result.stoppedReason, 'depth');
  assert.equal(result.score, 0);
  // These traces were established with eager stable sorting. Equal work and
  // PVs guard ordering, including tied moves and histories learned by siblings.
  assert.equal(result.nodes, nodes);
  assert.deepEqual(result.pv, pv);
  let current = position;
  for (const action of result.pv) current = validateAction(current, action);
}

test('deferred ordering preserves the eager stable order when bonuses tie', () => {
  const position = createPosition();
  const result = analyze(position, { ...limits, quiescenceDepth: 0,
    heuristics: { quietCentralization: 0, historyBonus: 0, killerBonus: 0, temporalMovePenalty: 0 } });
  verify(position, result, 1044, [
    [[[0, 0, 0, 6], [0, 0, 2, 5]]],
    [[[0, 1, 7, 6], [0, 1, 5, 5]]],
  ]);
});

test('suspended multiboard ordering preserves fractional priorities across sibling searches', () => {
  const position = createPosition({ variant: 'two_timelines' });
  const result = analyze(position, { ...limits, quiescenceDepth: 1,
    heuristics: { quietCentralization: 10.125, historyBonus: 20.375,
      killerBonus: 100000.125, temporalMovePenalty: 2000000.25 } });
  verify(position, result, 23579, [
    [[[1, 0, 0, 6], [1, 0, 2, 5]], [[2, 0, 0, 6], [2, 0, 2, 5]]],
    [[[2, 1, 7, 6], [2, 1, 5, 5]], [[1, 1, 7, 6], [1, 1, 5, 5]]],
  ]);
});

test('compact quiet history preserves the baseline search on 16 by 16 boards', () => {
  const squares = Array.from({ length: 16 }, () => Array(16).fill(0));
  squares[0][0] = 12; squares[15][15] = 11;
  squares[1][14] = 4; squares[14][1] = 3;
  squares[1][5] = 2; squares[14][10] = 1;
  const position = { board: [[squares]], action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3] };
  const original = structuredClone(position);
  const result = analyze(position, { ...limits, maxDepth: 3, quiescenceDepth: 0,
    heuristics: { quietCentralization: 10.125, historyBonus: 20.375, killerBonus: 100000.125 } });
  // Recorded with string history keys. High ranks/files, captures and quiet
  // replies must keep distinct histories even with fractional bonuses.
  assert.equal(result.depth, 3);
  assert.equal(result.stoppedReason, 'depth');
  assert.equal(result.score, 482);
  // The searched order stays identical; unordered single-timeline legality
  // witnesses perform one additional rejected move probe (two work ticks).
  assert.equal(result.nodes, 1917);
  assert.deepEqual(result.pv, [
    [[[0, 0, 1, 14], [0, 0, 14, 1]]],
    [[[0, 1, 14, 10], [0, 1, 13, 10]]],
    [[[0, 2, 14, 1], [0, 2, 8, 7]]],
  ]);
  let current = position;
  for (const action of result.pv) current = validateAction(current, action);
  assert.deepEqual(position, original);
});
