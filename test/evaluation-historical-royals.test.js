import test from 'node:test';
import assert from 'node:assert/strict';
import { createEvaluator, evaluateDetailed, inspectEvaluation } from '../src/evaluate.js';
import { raw } from '../src/rules.js';

function distantRoyal({ first = 0, distance = 7, piece = 4 } = {}) {
  const latest = first + distance * 2;
  const timeline = Array.from({ length: latest + 1 }, (_, t) => {
    if (t < first) return null;
    const squares = Array.from({ length: 8 }, () => Array(8).fill(0));
    squares[0][7] = 12;
    squares[7][7] = 11;
    return squares;
  });
  const targetRank = piece === 4 ? 7 : 0;
  timeline[first][7][7] = 0;
  timeline[first][targetRank][0] = 11;
  timeline[latest][0][0] = piece;
  return { board: [timeline], action: 0 };
}

const attacks = (position, source, target) => raw.pieceFuncs.moves(position.board, source).some(move =>
  target.every((coordinate, index) => move[1][index] === coordinate));

test('earliest royal copies remain temporal targets beyond the recent-history window', () => {
  for (const options of [{}, { first: 4 }, { distance: 20, piece: 8 }]) {
    const position = distantRoyal(options);
    const first = options.first || 0, latest = position.board[0].length - 1;
    const targetRank = options.piece === 8 ? 0 : 7;
    assert(attacks(position, [0, latest, 0, 0], [0, first, targetRank, 0]));
    assert.equal(evaluateDetailed(position).temporal, 13,
      'The oldest king retains the normal discounted historical pressure.');
  }
});

test('long-range pressure still stops at occupied or missing intermediate history', () => {
  for (const blocker of [1, 2, null]) {
    const position = distantRoyal();
    if (blocker === null) position.board[0][8] = null;
    else position.board[0][8][3][0] = blocker;
    assert.equal(attacks(position, [0, 14, 0, 0], [0, 0, 7, 0]), false);
    assert.equal(evaluateDetailed(position).temporal, 0);
  }
});

test('an oldest royal on the opposite half-turn cannot create temporal pressure', () => {
  const position = distantRoyal();
  position.board[0][0][7][0] = 0;
  position.board[0][1][7][0] = 11;
  assert.equal(attacks(position, [0, 14, 0, 0], [0, 1, 7, 0]), false);
  assert.equal(evaluateDetailed(position).temporal, 0);
});

test('oldest-royal pressure is color-symmetric and agrees with cached immutable evaluation', () => {
  const white = distantRoyal();
  const black = {
    action: 1,
    board: white.board.map(line => [null, ...line.map(squares => squares.toReversed().map(row => row.map(piece =>
      !piece ? 0 : piece + (piece % 2 ? 1 : -1))))]),
  };
  assert(attacks(black, [0, 15, 7, 0], [0, 1, 0, 0]));
  assert.equal(evaluateDetailed(black).temporal, -evaluateDetailed(white).temporal);
  const freeze = value => {
    if (!value || typeof value !== 'object' || Object.isFrozen(value)) return;
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  };
  const cached = createEvaluator();
  for (const position of [white, black, white]) {
    const original = JSON.stringify(position);
    freeze(position);
    const result = evaluateDetailed(position);
    assert.equal(cached(position), result.total);
    assert.equal(inspectEvaluation(position).total, result.total);
    assert.equal(JSON.stringify(position), original);
  }
});
