import test from 'node:test';
import assert from 'node:assert/strict';
import { createSearchSession, MATE_SCORE } from '../src/search.js';
import { createPosition, validateAction } from '../src/rules.js';

const limits = { unlimitedTime: true, maxNodes: 100000, quiescenceDepth: 2 };
const request = (position, horizon, extra = {}) => ({
  position, horizon, remaining: 0, ply: 1, alpha: -1000000, beta: 1000000, ...extra,
});

test('quiet warmup reuses legal-turn proofs across cloned positions and tactical horizons', () => {
  const position = createPosition();
  const warmed = createSearchSession(position, limits);
  const quiet = warmed.subtree(request(position, 0));
  const before = warmed.statistics();
  const deepened = warmed.subtree(request(structuredClone(position), 2));
  const after = warmed.statistics();
  const fresh = createSearchSession(position, limits);
  const reference = fresh.subtree(request(position, 2));
  assert.deepEqual(deepened, reference);
  assert.equal(deepened.score, quiet.score);
  assert(after.generationNodes - before.generationNodes < fresh.statistics().generationNodes,
    'a known legal position must not regenerate and discard another quiet turn');

  // A narrow-window visit to the same quiet position can use the exact value
  // proved by exhausting the tactical actions, even outside the first window.
  const bounded = createSearchSession(position, limits);
  const upper = bounded.subtree(request(position, 2, { alpha: quiet.score + 10, beta: quiet.score + 11 }));
  const generated = bounded.statistics().generationNodes;
  const exact = bounded.subtree(request(position, 2));
  assert.equal(exact.score, upper.score);
  assert.equal(bounded.statistics().generationNodes, generated);
});

test('warmup facts never replace recapture search or leak into different histories', () => {
  const position = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  const poisoned = validateAction(position, [[[0, 4, 2, 5], [0, 4, 4, 4]]]);
  const warmed = createSearchSession(poisoned, limits);
  const quiet = warmed.subtree(request(poisoned, 0));
  const tactical = warmed.subtree(request(structuredClone(poisoned), 1));
  const fresh = createSearchSession(poisoned, { ...limits, maxTableEntries: 0 });
  const reference = fresh.subtree(request(poisoned, 1));
  assert.equal(tactical.score, reference.score);
  assert.notEqual(tactical.score, quiet.score, 'Black must search the recapture of the knight');
  let current = poisoned;
  for (const action of tactical.pv) current = validateAction(current, action);

  const changed = structuredClone(poisoned);
  changed.board[0][5][4][4] = 0;
  const changedResult = warmed.subtree(request(changed, 1));
  const changedReference = createSearchSession(changed, limits).subtree(request(changed, 1));
  assert.deepEqual(changedResult, changedReference);
});

test('cached empty full-turn trees preserve mate distance and avoid repeated generation', () => {
  const position = createPosition({ pgn: '1. e3 / f6 2. Qe2 / Nc6 3. Qh5' });
  const session = createSearchSession(position, limits);
  const first = session.subtree(request(position, 2, { remaining: 2 }));
  const generated = session.statistics().generationNodes;
  const repeated = session.subtree(request(structuredClone(position), 2, { remaining: 2, ply: 3 }));
  assert.equal(first.score, -MATE_SCORE + 1);
  assert.equal(repeated.score, -MATE_SCORE + 3);
  assert.deepEqual(repeated.pv, []);
  assert.equal(session.statistics().generationNodes, generated);
});

test('cached check facts still require evasions when the tactical horizon grows', () => {
  const position = {
    board: [[[[5, 0, 0, 0], [10, 12, 0, 9], [11, 0, 0, 0], [0, 0, 0, 0]]]],
    action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3],
  };
  const session = createSearchSession(position, limits);
  session.subtree(request(position, 0));
  const deepened = session.subtree(request(structuredClone(position), 1));
  const reference = createSearchSession(position, { ...limits, maxTableEntries: 0 }).subtree(request(position, 1));
  assert.equal(deepened.score, reference.score);
  assert(deepened.pv.length > 0, 'a checked position cannot stand pat');
  let current = position;
  for (const action of deepened.pv) current = validateAction(current, action);
});

test('an unrestricted policy-boundary witness cannot bypass later restricted legality checks', () => {
  const future = [[0, 8, 0], [11, 0, 0], [4, 0, 12]];
  const position = {
    board: [[[[0, 11, 12], [7, 0, 0], [3, 0, 0]]], null,
      Array.from({ length: 3 }, () => structuredClone(future))],
    action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3],
  };
  const session = createSearchSession(position, limits);
  session.subtree(request(position, 0));
  const before = session.statistics();
  assert.equal(before.policyLeaves, 1);
  const result = session.subtree(request(structuredClone(position), -1));
  const after = session.statistics();
  const reference = createSearchSession(position, { ...limits, maxTableEntries: 0 }).subtree(request(position, -1));
  assert.deepEqual(result, reference);
  assert.deepEqual(result.pv, []);
  assert.equal(after.policyLeaves, 2);
  assert(after.generationNodes > before.generationNodes,
    'the optional-board witness must not be reused as a permitted turn');
});
