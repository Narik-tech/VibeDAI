import test from 'node:test';
import assert from 'node:assert/strict';
import { analyze, MATE_SCORE } from '../src/search.js';
import { applyMove, createPosition, generateActions, inCheck, raw, submitPosition, validateAction } from '../src/rules.js';

const limits = { timeMs: 10000, quiescenceDepth: 2 };

function validatePolicyPv(position, result) {
  let current = position;
  for (const action of result.pv) {
    for (const move of action) {
      const [from, to] = move;
      if (from[0] === to[0] && from[1] === to[1]) {
        assert(raw.boardFuncs.present(current.board, current.action).includes(from[0]),
          'ordinary moves must originate on the present at the time they are played');
      }
      current = applyMove(current, move);
    }
    current = submitPosition(current);
  }
}

test('mate-distance bounds finish a verified immediate mate within 250 work nodes', () => {
  const position = createPosition({ pgn: '1. e3 / f6 2. Qe2 / Nc6' });
  for (const maxTableEntries of [0, 100000]) {
    const result = analyze(position, { ...limits, maxDepth: 2, maxNodes: 250, maxTableEntries });
    assert.equal(result.completed, true);
    assert.equal(result.stoppedReason, 'mate');
    assert.equal(result.depth, 1);
    assert.equal(result.score, MATE_SCORE - 1);
    assert.equal(result.mateIn, 1);
    const after = validateAction(position, result.bestAction);
    assert(inCheck(after));
    // Verify with all legal actions, independently of the search policy.
    assert.deepEqual([...generateActions(after)], []);
    validatePolicyPv(position, result);
  }
});

test('present-spatial search reaches opening depth four within 20000 work nodes', () => {
  const position = createPosition();
  // Timeline reserve/entry scores change the explored tree. Keep a fixed work
  // ceiling with modest headroom for evaluation changes, retaining full depth.
  const result = analyze(position, { ...limits, maxDepth: 4, maxNodes: 20000 });
  assert.equal(result.depth, 4);
  assert.equal(result.effectiveQuiescenceDepth, 2);
  assert.equal(result.stoppedReason, 'depth');
  assert.equal(result.searchPolicy, 'present-spatial');
  validatePolicyPv(position, result);
});

test('two-timeline search finishes depth two with recaptures within 25000 work nodes', () => {
  const position = createPosition({ variant: 'two_timelines' });
  const result = analyze(position, { ...limits, maxDepth: 2, quiescenceDepth: 1, maxNodes: 25000 });
  assert.equal(result.depth, 2);
  assert.equal(result.effectiveQuiescenceDepth, 1);
  assert.equal(result.stoppedReason, 'depth');
  assert.equal(result.score, 0);
  validatePolicyPv(position, result);
});

test('aspiration and tactical witnesses complete the locked-king puzzle within 5000 work nodes', () => {
  const position = createPosition({ pgn: '[Board "Custom"]\n[k7/pn6/K7/8/8/8/6PB/8:0:1:w]' });
  const result = analyze(position, { ...limits, maxDepth: 3, maxNodes: 5000 });
  assert.equal(result.depth, 3);
  assert.equal(result.effectiveQuiescenceDepth, 2);
  assert.equal(result.stoppedReason, 'depth');
  assert.equal(result.scoreType, 'cp');
  assert.deepEqual(result.bestAction, [[[0, 0, 1, 7], [0, 0, 0, 6]]]);
  validatePolicyPv(position, result);
});

test('search PV and progress snapshots never include ordinary moves on future or inactive boards', () => {
  const board = [[12, 0, 0, 0], [8, 0, 9, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
  const position = {
    board: [[structuredClone(board)], null,
      Array.from({ length: 3 }, () => structuredClone(board)), null, [structuredClone(board)]],
    action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3],
  };
  const result = analyze(position, {
    ...limits, maxDepth: 2, maxNodes: 10000,
    onProgress: progress => validatePolicyPv(position, progress),
  });
  assert(result.bestAction);
  validateAction(position, result.bestAction);
  validatePolicyPv(position, result);
});

test('exhausting the spatial policy does not invent mate or select an excluded fallback', () => {
  const future = [[0, 8, 0], [11, 0, 0], [4, 0, 12]];
  const position = {
    board: [[[[0, 11, 12], [7, 0, 0], [3, 0, 0]]], null,
      Array.from({ length: 3 }, () => structuredClone(future))],
    action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3],
  };
  // Moving the optional rook first makes its source historical. The present
  // king can then escape by jumping there and creating a different branch.
  // That legal escape is deliberately outside the requested search policy.
  const legalEscape = [[[2, 2, 0, 1], [2, 2, 0, 2]], [[0, 0, 0, 2], [2, 2, 0, 2]]];
  validateAction(position, legalEscape);
  assert(inCheck(position));
  assert.deepEqual([...generateActions(position, { skipOptionalSpatial: true })], []);
  const result = analyze(position, { ...limits, maxDepth: 3, maxNodes: 10000 });
  assert.equal(result.status, 'incomplete');
  assert.equal(result.completed, false);
  assert.equal(result.stoppedReason, 'policy');
  assert.equal(result.policyLeaves, 1);
  assert.equal(result.searchPolicy, 'present-spatial');
  assert.equal(result.score, null);
  assert.equal(result.scoreType, 'unavailable');
  assert.equal(result.mateIn, null);
  assert.equal(result.bestAction, null);
  assert.deepEqual(result.pv, []);
});
