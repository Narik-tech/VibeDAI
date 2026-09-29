import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateDetailed } from '../src/evaluate.js';
import { raw } from '../src/rules.js';

const empty = () => Array.from({ length: 8 }, () => Array(8).fill(0));
function kings() {
  const squares = empty();
  squares[0][4] = 12;
  squares[7][4] = 11;
  return squares;
}
function timelines(indices) {
  const board = [];
  for (const line of indices) board[line] = [kings()];
  return { board, action: 0 };
}
const reserveScore = indices => evaluateDetailed(timelines(indices)).timelines;
const travelScore = position => evaluateDetailed(position).travel;
const attacks = (position, source, target) => raw.pieceFuncs.moves(position.board, source).some(move =>
  target.every((coordinate, index) => move[1][index] === coordinate));

function opportunity() {
  const timeline = Array.from({ length: 5 }, kings);
  // White's queen on f5 can capture Black's historical f7 pawn, defended
  // only by the adjacent e8 king, along the time/rank diagonal through f6.
  timeline[4][4][5] = 10;
  timeline[0][6][5] = 1;
  return { board: [timeline], action: 0 };
}
const source = [0, 4, 4, 5], target = [0, 0, 6, 5];

test('timeline reserves favor the player who has created fewer branches', () => {
  assert.equal(reserveScore([0]), 0);
  assert.equal(reserveScore([0, 1, 2]), 0);
  assert(reserveScore([0, 2]) < 0, 'White has spent its immediately active branch reserve.');
  assert(reserveScore([0, 1]) > 0, 'Black has spent its immediately active branch reserve.');
  assert.equal(reserveScore([0, 1]), -reserveScore([0, 2]));
  assert.equal(reserveScore([0, 1, 3]), -reserveScore([0, 2, 4]));
});

test('inactive overextension costs more than spending a single timeline reserve', () => {
  const position = timelines([0, 2, 4]);
  assert.equal(raw.boardFuncs.active(position.board).includes(4), false);
  assert(evaluateDetailed(position).timelines < reserveScore([0, 2]));
});

test('even starting timelines are balanced and sparse layouts use timeline extents', () => {
  assert.equal(raw.boardFuncs.isEvenTimeline(timelines([1, 2]).board), true);
  assert.equal(reserveScore([1, 2]), 0);
  assert(reserveScore([1, 2, 4]) < 0);
  assert.equal(reserveScore([1, 2, 3]), -reserveScore([1, 2, 4]));
  // Missing inner branches do not refund the capacity already consumed by
  // the outermost timeline's coordinate.
  assert.equal(reserveScore([0, 6]), reserveScore([0, 2, 4, 6]));
  assert.equal(reserveScore([0, 5]), reserveScore([0, 1, 3, 5]));
});

test('pawn entry routes preserve directional adjacency across even and sparse timelines', () => {
  for (const [color, sourceLine, targetLine, expected] of [
    [0, 0, 1, 41], [0, 0, 3, 0], [0, 2, 1, 41], [0, 4, 1, 0],
    [1, 0, 2, -41], [1, 0, 4, 0], [1, 1, 2, -41], [1, 3, 2, 0],
  ]) {
    const sourceTime = 2 + color, targetTime = color;
    const board = [], origin = empty(), target = empty(), latest = empty();
    origin[3][3] = color === 0 ? 2 : 1;
    origin[color === 0 ? 0 : 7][0] = color === 0 ? 12 : 11;
    target[3][3] = color === 0 ? 1 : 2;
    target[3][4] = latest[3][4] = color === 0 ? 11 : 12;
    board[sourceLine] = [];
    board[sourceLine][sourceTime] = origin;
    board[targetLine] = [];
    board[targetLine][targetTime] = target;
    board[targetLine][targetTime + 1] = latest;
    const position = { board, action: color };
    assert.equal(travelScore(position), expected, `color ${color}: ${sourceLine} -> ${targetLine}`);
  }
});

test('an open historical capture of a king-adjacent pawn is a strong travel opportunity', () => {
  const position = opportunity();
  assert(attacks(position, source, target));
  assert(travelScore(position) >= 100, 'A concrete undefended king-zone entry should be worth at least a pawn.');
});

test('historical blockers and missing intermediate boards close travel opportunities', () => {
  const clear = opportunity();
  for (const blocker of [2, 1, null]) {
    const position = structuredClone(clear);
    if (blocker === null) position.board[0][2] = null;
    else position.board[0][2][5][5] = blocker;
    assert.equal(attacks(position, source, target), false);
    assert.equal(travelScore(position), 0);
    assert.equal(evaluateDetailed(position).material, evaluateDetailed(clear).material);
  }
  const otherHalfTurn = structuredClone(clear);
  otherHalfTurn.board[0][1][5][5] = 2;
  assert(attacks(otherHalfTurn, source, target));
  assert.equal(travelScore(otherHalfTurn), travelScore(clear));
});

test('the travel opportunity requires an available timeline reserve', () => {
  const position = opportunity();
  position.board[2] = Array.from({ length: 5 }, kings);
  assert(attacks(position, source, target), 'The geometric capture still exists after spending the reserve.');
  assert.equal(travelScore(position), 0);
  // A Black branch balances the extents and restores White's capacity.
  position.board[1] = Array.from({ length: 5 }, kings);
  assert(travelScore(position) > 0);
});

test('nonroyal defenders remove the undefended king-zone pawn bonus', () => {
  const position = opportunity();
  position.board[0][0][7][6] = 3; // Black bishop on g8 protects f7.
  assert(attacks(position, source, target));
  assert.equal(travelScore(position), 0);
});

test('an undefended historical pawn far from its king does not receive the king-zone bonus', () => {
  const position = opportunity();
  position.board[0][0][7][4] = 0;
  position.board[0][0][7][0] = 11;
  assert(attacks(position, source, target));
  assert.equal(travelScore(position), 0);
});

test('ordinary captures and obsolete historical attackers do not create travel bonuses', () => {
  const ordinary = opportunity();
  ordinary.board[0][0][6][5] = 0;
  ordinary.board[0][4][6][5] = 1;
  assert(attacks(ordinary, source, [0, 4, 6, 5]));
  assert.equal(travelScore(ordinary), 0);

  const historical = opportunity();
  historical.board[0][4][4][5] = 0;
  historical.board[0][2][5][5] = 10;
  assert(attacks(historical, [0, 2, 5, 5], target));
  assert.equal(travelScore(historical), 0);
});

test('travel captures cannot connect opposite half-turn colors', () => {
  const position = opportunity();
  position.board[0][0][6][5] = 0;
  position.board[0][1][6][5] = 1;
  assert.equal(attacks(position, source, [0, 1, 6, 5]), false);
  assert.equal(travelScore(position), 0);
});

test('a route prepared for the next half-turn is discounted and still respects recorded blockers', () => {
  const ready = opportunity();
  const setup = structuredClone(ready);
  setup.board[0].pop();
  setup.board[0][3][4][5] = 10;
  setup.action = 1;
  assert.equal(attacks(setup, [0, 3, 4, 5], target), false);
  assert(travelScore(setup) > 0);
  assert(travelScore(setup) < travelScore(ready));
  setup.board[0][2][5][5] = 2;
  assert.equal(travelScore(setup), 0);
});

test('multiple attackers and copied timelines do not multiply a single travel opportunity', () => {
  const single = opportunity();
  const multiple = structuredClone(single);
  multiple.board[0][4][4][3] = 10;
  assert(attacks(multiple, [0, 4, 4, 3], target));
  assert.equal(travelScore(multiple), travelScore(single));
  multiple.board[1] = structuredClone(single.board[0]);
  multiple.board[2] = structuredClone(single.board[0]);
  assert.equal(travelScore(multiple), travelScore(single));
});

test('Black receives the same travel opportunity with colors and move parity reflected', () => {
  const white = opportunity();
  const black = {
    action: 1,
    board: white.board.map(line => [null, ...line.map(squares => squares.toReversed().map(row => row.map(piece =>
      !piece ? 0 : piece + (piece % 2 ? 1 : -1))))]),
  };
  assert(attacks(black, [0, 5, 3, 5], [0, 1, 1, 5]));
  assert.equal(travelScore(black), -travelScore(white));
});

test('timeline and travel evaluation preserve frozen historical state', () => {
  const position = opportunity();
  const original = JSON.stringify(position);
  const freeze = value => {
    if (!value || typeof value !== 'object' || Object.isFrozen(value)) return;
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  };
  freeze(position);
  const result = evaluateDetailed(position);
  assert(result.travel > 0);
  assert.equal(JSON.stringify(position), original);
});
