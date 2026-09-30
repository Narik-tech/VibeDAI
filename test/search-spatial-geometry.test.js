import test from 'node:test';
import assert from 'node:assert/strict';
import { applyMove, createSearchMoveGenerator, pseudoMoves, raw } from '../src/rules.js';

const empty = (height = 8, width = 8) => Array.from({ length: height }, () => Array(width).fill(0));
const promotions = [10, 9, 8, 7, 6, 5, 4, 3];
const freeze = value => {
  if (Array.isArray(value)) { for (const item of value) freeze(item); Object.freeze(value); }
  return value;
};

test('search king generation preserves castling order, blockers, flags and every attacked king square', () => {
  for (const color of [0, 1]) for (const width of [8, 16]) {
    const rank = color ? 7 : 0, file = width === 8 ? 4 : 7;
    const base = empty(8, width);
    base[rank][file] = -(12 - color);
    base[rank][0] = base[rank][width - 1] = -(8 - color);
    const fixtures = [
      ['clear', squares => {}, [1, -1]],
      ['moved king', squares => { squares[rank][file] *= -1; }, []],
      ['origin attacked', squares => { squares[4][file] = 7 + color; }, []],
    ];
    for (const direction of [1, -1]) {
      const rook = direction === 1 ? width - 1 : 0;
      const otherSide = [-direction];
      fixtures.push(
        [`${direction} moved rook`, squares => { squares[rank][rook] *= -1; }, otherSide],
        [`${direction} wrong-color rook`, squares => { squares[rank][rook] = -(7 + color); }, []],
        [`${direction} blocked`, squares => { squares[rank][file + direction] = 4 - color; }, otherSide],
      );
      for (const offset of [1, 2]) fixtures.push([
        `${direction} attacked offset ${offset}`,
        squares => { squares[4][file + offset * direction] = 7 + color; }, otherSide,
      ]);
    }
    for (const [label, edit, directions] of fixtures) {
      const squares = structuredClone(base);
      edit(squares);
      const timeline = color ? [null, squares] : [squares];
      const position = { board: [timeline], action: color, promotions };
      freeze(position.board);
      const actual = createSearchMoveGenerator()(position);
      assert.deepEqual(actual, pseudoMoves(position), `${label}, color ${color}, width ${width}`);
      assert.deepEqual(actual.filter(move => move.length === 4).map(move => Math.sign(move[1][3] - file)),
        directions, `${label}: castle direction and order`);
    }
  }
});

test('spatial appends preserve upstream sparse timeline normalization and immutable long histories', () => {
  for (const source of [0, 1, 4, 8]) for (const turn of [0, 1, 2, 63, 512, 2047]) {
    const color = turn % 2, board = Array(source + 3), timeline = Array(turn + 1);
    if (source > 2) { board[0] = undefined; board[1] = null; board[2] = []; }
    const squares = empty();
    squares[2][2] = -(8 - color);
    squares[2][3] = 1 + color;
    timeline[turn] = squares;
    if (turn > 2) { timeline[0] = empty(); timeline[1] = undefined; timeline[2] = null; }
    board[source] = timeline;
    board[source + 2] = [empty()];
    const start = { board, action: color, promotions }, before = structuredClone(start);
    freeze(board);
    for (const destination of [[2, 1], [4, 2], [2, 3]]) {
      const move = [[source, turn, 2, 2], [source, turn, ...destination]];
      const expected = structuredClone(start);
      raw.boardFuncs.move(expected.board, move);
      const actual = applyMove(start, move);
      assert.deepEqual(actual, expected, `source ${source}, turn ${turn}, destination ${destination}`);
      assert.equal(actual.board[source][turn], squares, 'old boards remain shared');
      assert.equal(actual.board[source + 2], board[source + 2], 'other histories remain shared');
      assert.equal(actual.board[source][turn + 1][7], squares[7], 'untouched ranks remain shared');
    }
    assert.deepEqual(start, before, 'siblings preserve the frozen parent');
  }
});
