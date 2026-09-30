import test from 'node:test';
import assert from 'node:assert/strict';
import { createSearchSession } from '../src/search.js';
import { createPosition, inCheck, validateAction } from '../src/rules.js';

const limits = { unlimitedTime: true, maxNodes: 100000, quiescenceDepth: 2 };
const promotions = [10, 9, 8, 7, 6, 5, 4, 3];
const request = (position, horizon) => ({
  position, horizon, remaining: 0, ply: 1, alpha: -1000000, beta: 1000000,
});

function validatePv(position, result) {
  let current = position;
  for (const action of result.pv) current = validateAction(current, action);
}

test('exhausted quiet tactical trees are reused across cloned positions and horizons', () => {
  const position = createPosition();
  const session = createSearchSession(position, limits);
  const first = session.subtree(request(position, 1));
  const before = session.statistics();
  const deeper = session.subtree(request(structuredClone(position), 2));
  const after = session.statistics();
  const reference = createSearchSession(position, { ...limits, maxTableEntries: 0 })
    .subtree(request(position, 2));

  assert.deepEqual(deeper, reference);
  assert.deepEqual(deeper, first);
  assert.deepEqual(deeper.pv, []);
  assert.equal(after.generationNodes, before.generationNodes,
    'a proved absence of tactical turns must not be regenerated at a larger horizon');
  assert.equal(after.qnodes, before.qnodes + 1);
  assert.equal(after.qTtHits, before.qTtHits + 1);
});

test('quiet-horizon stand pat cannot hide a legal recapture at a larger horizon', () => {
  const opening = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  const position = validateAction(opening, [[[0, 4, 2, 5], [0, 4, 4, 4]]]);
  const session = createSearchSession(position, limits);
  const quiet = session.subtree(request(position, 0));
  const generated = session.statistics().generationNodes;
  const tactical = session.subtree(request(structuredClone(position), 1));
  const reference = createSearchSession(position, { ...limits, maxTableEntries: 0 })
    .subtree(request(position, 1));

  assert.equal(tactical.score, reference.score);
  assert.notEqual(tactical.score, quiet.score);
  assert(session.statistics().generationNodes > generated);
  assert(tactical.pv.length > 0, 'Black must search the recapture of the knight');
  validatePv(position, tactical);
});

test('checked horizons still search evasions after a cached depth-zero result', () => {
  const position = {
    board: [[[[5, 0, 0, 0], [10, 12, 0, 9], [11, 0, 0, 0], [0, 0, 0, 0]]]],
    action: 0, promotions,
  };
  assert(inCheck(position));
  const session = createSearchSession(position, limits);
  session.subtree(request(position, 0));
  const generated = session.statistics().generationNodes;
  const deeper = session.subtree(request(structuredClone(position), 1));
  const reference = createSearchSession(position, { ...limits, maxTableEntries: 0 })
    .subtree(request(position, 1));

  assert.equal(deeper.score, reference.score);
  assert(session.statistics().generationNodes > generated);
  assert(deeper.pv.length > 0, 'a checked position must search an evasion');
  validatePv(position, deeper);
});

test('an empty tactical iterator cannot turn stalemate into a static quiet score', () => {
  const position = { board: [[[[12, 0, 0], [0, 0, 11], [0, 9, 0]]]], action: 0, promotions };
  assert.equal(inCheck(position), false);
  const session = createSearchSession(position, limits);
  for (const horizon of [1, 2, 0]) {
    const result = session.subtree(request(structuredClone(position), horizon));
    assert.equal(result.score, 0);
    assert.deepEqual(result.pv, []);
  }
});
