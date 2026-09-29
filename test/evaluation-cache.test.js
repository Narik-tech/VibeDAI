import test from 'node:test';
import assert from 'node:assert/strict';
import { applyMove, createPosition, pseudoMoves } from '../src/rules.js';
import { createEvaluator, evaluate, evaluateDetailed, inspectEvaluation } from '../src/evaluate.js';

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

test('temporal target groups preserve mixed parity, projected travel, and weighted maxima', () => {
  const history = Array.from({ length: 19 }, (_, t) => {
    const squares = Array.from({ length: 8 }, () => Array(8).fill(0));
    squares[0][1] = 20;
    squares[0][3] = 12;
    squares[7][4] = 11;
    if (t === 14) squares[6][5] = 1;
    if (t === 15) squares[1][2] = 2; // Shared by White's two royal zones.
    return squares;
  });
  history[2] = null;
  history[9] = null;
  history[18][4][5] = 10; // Ready to capture the historical black pawn.
  history[18][3][2] = 9; // Prepares a capture of the white pawn next half-turn.
  history[18][7][4] = 8; // Pressures historical black kings.
  history[18][7][7] = 11;
  const profiles = [undefined, { inactiveWeight: 0.37, historicalPressureWeight: 0.23,
    temporalPressureWeight: 1.7, travelOpportunityWeight: 2.3, queenValue: 135 },
  { historicalPressureWeight: 0, travelOpportunityWeight: 0, temporalPressureWeight: 0 }];
  const boards = [[history], [history, history.slice(1), history],
    [history, history.slice(1), history, null, history]];
  // Literal [pressure, travel, total] scores recorded before target grouping.
  // The last layout spends White's reserve and includes an inactive frontier.
  const expected = [
    [[13, 70, 678], [8, 64, 701], [0, 0, 595]],
    [[39, 70, 770], [23, 64, 782], [0, 0, 661]],
    [[52, -70, 463], [31, -64, 482], [0, 0, 481]],
  ];
  for (let profile = 0; profile < profiles.length; profile++) {
    const cached = createEvaluator(profiles[profile]);
    for (let layout = 0; layout < boards.length; layout++) {
      const position = { board: boards[layout], action: 0 };
      const result = evaluateDetailed(position, profiles[profile]);
      assert.deepEqual([result.temporal, result.travel, result.total], expected[layout][profile]);
      assert.equal(cached(position), result.total);
      assert.equal(inspectEvaluation(position, profiles[profile]).total, result.total);
    }
  }
});
