import test from 'node:test';
import assert from 'node:assert/strict';
import { applyMove, canSubmit, createPosition, createSearchMoveGenerator, generateActions, pseudoMoves } from '../src/rules.js';

const empty = (height = 5, width = height) => Array.from({ length: height }, () => Array(width).fill(0));
const freeze = value => {
  if (Array.isArray(value)) { for (const child of value) freeze(child); Object.freeze(value); }
  return value;
};

function compare(position, label) {
  const original = structuredClone(position);
  freeze(position.board);
  freeze(position.promotions);
  const actual = createSearchMoveGenerator()(position);
  assert.deepEqual(actual, pseudoMoves(position), label);
  assert.deepEqual(position, original, `${label}: preserve the input`);
  return actual;
}

test('pawn generation retains both en-passant directions, colors and brawn history', () => {
  for (const color of [0, 1]) for (const movingType of [1, 8]) {
    for (const capturedType of [1, 8]) for (const historicalType of [1, 8]) {
      const turn = 2 + color, rank = color ? 2 : 3, forward = color ? -1 : 1;
      const board = [Array.from({ length: turn + 1 }, () => empty(6))];
      board[0][turn][rank][2] = movingType * 2 - color;
      for (const file of [1, 3]) {
        board[0][turn][rank][file] = capturedType * 2 - 1 + color;
        board[0][turn - 2][rank + 2 * forward][file] = -(historicalType * 2 - 1 + color);
      }
      const moves = compare({ board, action: color, promotions: [10, 9] },
        `en passant: color ${color}, types ${movingType}/${capturedType}/${historicalType}`);
      assert.equal(moves.filter(move => move.length === 3).length, 2);
    }
  }
  const position = createPosition({ pgn: '1. e4 / a6 2. e5 / d5' });
  assert(compare(position, 'recorded en passant').some(move => move.length === 3));
});

test('pawn generation retains promotion order and all sparse temporal geometry', () => {
  let seed = 0x9a75c;
  const random = limit => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed % limit;
  };
  for (let fixture = 0; fixture < 96; fixture++) {
    const height = 3 + random(5), width = 3 + random(5), color = fixture % 2;
    const source = 1 + random(5), turn = 2 + 2 * random(3) + color;
    const board = Array.from({ length: 7 }, (_, line) => {
      if (line !== source && random(4) === 0) return null;
      return Array.from({ length: turn + 3 }, (_, time) => {
        if ((line !== source || time !== turn) && random(4) === 0) return null;
        const squares = empty(height, width);
        for (const [rank, row] of squares.entries()) for (let file = 0; file < row.length; file++) {
          if (random(5) !== 0) continue;
          let piece = 3 + random(22);
          if ([15, 16].includes(piece) && rank === (piece % 2 ? 0 : height - 1)) piece = 4 - piece % 2;
          row[file] = (random(2) ? -1 : 1) * piece;
        }
        return squares;
      });
    });
    board[source].length = turn + 1;
    for (let file = 0; file < width; file++) {
      const rank = color ? 1 : height - 2;
      const type = random(2) ? 1 : 8;
      board[source][turn][rank][file] = (random(2) ? -1 : 1) * (type * 2 - color);
    }
    const promotions = [[6, 5, 10, 9, 20, 19], [4, 3], [], undefined][fixture % 4];
    compare({ board, action: color, promotions }, `promotion/temporal fixture ${fixture}`);
  }
});

test('en-passant geometry still lets action validation reject an exposed royal', () => {
  for (const exposed of [false, true]) {
    const past = empty(8), latest = empty(8);
    for (const squares of [past, latest]) {
      squares[7][0] = 11;
      squares[4][7] = 12;
      squares[4][6] = 2;
      squares[exposed ? 4 : 5][0] = 7;
    }
    past[6][5] = -1;
    latest[4][5] = 1;
    const position = { board: [[past, null, latest]], action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3] };
    const moves = compare(position, `en-passant royal exposure ${exposed}`);
    const capture = moves.find(move => move.length === 3);
    assert(capture);
    assert.equal(canSubmit(applyMove(position, capture)), !exposed);
    const generated = [...generateActions(position, { generateMoves: createSearchMoveGenerator() })];
    assert.deepEqual(generated, [...generateActions(position)]);
    assert.equal(generated.some(action => action.moves.some(move => move.length === 3)), !exposed);
  }
});
