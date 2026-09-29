import test from 'node:test';
import assert from 'node:assert/strict';
import { applyMove, createPosition, pseudoMoves } from '../src/rules.js';
import { createEvaluator, evaluate, inspectEvaluation } from '../src/evaluate.js';

test('search evaluation reuses immutable board facts across sibling histories and profiles', () => {
  for (const setup of [{}, { variant: 'two_timelines' }, { pgn: '1. e4 / e5 2. Nf3 / Nc6' }]) {
    const position = createPosition(setup), original = structuredClone(position);
    const positions = [position, ...pseudoMoves(position).map(move => applyMove(position, move))];
    for (const heuristics of [undefined, { historicalPressureWeight: 0.4, inactiveWeight: 0.15,
      corridorWeight: 1.7, travelOpportunityWeight: 0.7, pawnValue: 120 }]) {
      const cached = createEvaluator(heuristics);
      for (const current of [...positions, ...positions.toReversed()]) {
        const expected = evaluate(current, heuristics);
        assert.equal(cached(current), expected);
        assert.equal(inspectEvaluation(current, heuristics).total, expected);
      }
    }
    assert.deepEqual(position, original);
  }
});

test('cached board facts retain current timeline coordinates, history and active weights', () => {
  const original = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  const timeline = original.board[0];
  const cached = createEvaluator();
  // Share the very same snapshots at different times and on active/inactive
  // branches. Only board-local king and pawn facts may cross these positions.
  const histories = [[timeline], [timeline, null, timeline],
    [timeline, null, timeline, null, timeline], [null, timeline, timeline],
    [[...timeline, ...timeline]], [timeline.slice(0, 3), null, timeline]];
  for (const board of [...histories, ...histories.toReversed()]) {
    const position = { ...original, board };
    assert.equal(cached(position), evaluate(position));
  }
});

test('public evaluation and fresh search caches see caller mutations', () => {
  const position = createPosition();
  const before = createEvaluator()(position);
  position.board[0][0][1][5] = 0;
  const after = evaluate(position);
  assert.notEqual(after, before);
  assert.equal(createEvaluator()(position), after);
  assert.equal(inspectEvaluation(position).total, after);
});
