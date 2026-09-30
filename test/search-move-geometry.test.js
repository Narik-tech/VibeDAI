import test from 'node:test';
import assert from 'node:assert/strict';
import { createPosition, createSearchMoveGenerator, pseudoMoves, raw } from '../src/rules.js';

const promotions = [24, 23, 20, 19, 10, 9, 8, 7, 6, 5, 4, 3];
const empty = (height = 5, width = height) => Array.from({ length: height }, () => Array(width).fill(0));
const position = (board, action = 0) => ({ board, action, promotions });

function compare(start, label = '') {
  const original = structuredClone(start);
  const generate = createSearchMoveGenerator();
  const expected = pseudoMoves(start), actual = generate(start);
  assert.deepEqual(actual, expected, `${label}: move coordinates and order must match the pinned library`);
  assert.equal(generate(start), actual, `${label}: repeated immutable geometry is reused`);
  assert.deepEqual(start, original, `${label}: move generation must preserve the input`);
  return actual;
}

function universe(color, layout) {
  const board = Array.from({ length: 7 }, () => Array.from({ length: 7 + color }, () => empty()));
  // The moving timeline ends earlier, allowing jumps into other timelines'
  // past, present and future. Timeline indices cross both sides of zero.
  board[2].length = 5 + color;
  if (layout === 'even') board[0] = null;
  if (layout === 'sparse') {
    delete board[3];
    board[0][2 + color] = null;
    delete board[1][4 + color];
    board[2][2 + color] = null;
    board[4].length = 3 + color;
    board[5][6 + color] = undefined;
  }
  return board;
}

test('every nonpawn geometry preserves direction order for both colors, flags and timeline layouts', () => {
  for (const type of [2, 3, 4, 5, 6, 7, 9, 10, 11, 12]) {
    for (const color of [0, 1]) for (const sign of [1, -1]) {
      for (const layout of ['odd', 'even', 'sparse']) {
        const board = universe(color, layout), turn = 4 + color;
        const piece = sign * (type * 2 - color);
        board[2][turn][2][2] = piece;
        const moves = compare(position(board, color), `piece ${piece}, ${layout}`);
        assert(moves.length > 0, `piece ${piece} needs actual destinations in the fixture`);
        assert(moves.some(move => move[1][0] !== 2 || move[1][1] !== turn),
          `piece ${piece} must exercise temporal geometry`);
      }
    }
  }
});

test('single-timeline direction tables preserve initial, historical and sparse move geometry', () => {
  for (const type of [2, 3, 4, 5, 6, 7, 9, 10, 11, 12]) {
    for (const color of [0, 1]) for (const turn of [color, 2 + color, 6 + color]) {
      const timeline = Array.from({ length: turn + 1 }, () => empty());
      timeline[turn][2][2] = type * 2 - color;
      const start = position([timeline], color);
      const moves = compare(start, `single timeline, type ${type}, turn ${turn}`);
      if (type === 3 || type === 4) assert(moves.length > 0, 'knights and rooks retain spatial moves');
      if (type === 4 && turn >= 2) assert(moves.some(move => move[1][1] < turn), 'time rays stay available');
      assert(moves.every(move => move[1][0] === 0));
      if (turn < 2) assert(moves.every(move => move[1][1] === turn));
      if (turn >= 2) {
        delete timeline[turn - 2];
        compare(start, `single timeline, type ${type}, missing historical board`);
      }
    }
  }
});

test('step captures, friendly blockers and royal exclusions retain their exact order', () => {
  for (const color of [0, 1]) for (const sign of [1, -1]) {
    const squares = empty(7), turn = color;
    const knight = sign * (6 - color), friendly = -(2 - color), enemy = -(1 + color);
    squares[3][3] = knight;
    squares[5][4] = friendly;
    squares[4][5] = enemy;
    squares[2][5] = 11 + color;
    squares[3][4] = friendly; // An intervening square cannot block a knight.
    const board = [Array.from({ length: turn + 1 }, () => empty(7))];
    board[0][turn] = squares;
    const moves = compare(position(board, color), `knight ${knight}`)
      .filter(move => move[0][2] === 3 && move[0][3] === 3);
    assert(moves.some(move => move[1][2] === 4 && move[1][3] === 5));
    assert(!moves.some(move => move[1][2] === 5 && move[1][3] === 4));
    assert(!moves.some(move => move[1][2] === 2 && move[1][3] === 5));
  }
});

test('rays stop at friendly pieces, captures, royals and missing historical boards', () => {
  for (const color of [0, 1]) for (const sign of [1, -1]) {
    const turn = 4 + color, board = universe(color, 'sparse');
    const squares = board[2][turn];
    squares[2][2] = sign * (8 - color);
    squares[2][3] = -(2 - color);
    squares[3][2] = -(1 + color);
    squares[1][2] = -(11 + color);
    const moves = compare(position(board, color), `rook ${squares[2][2]}`)
      .filter(move => move[0][0] === 2 && move[0][2] === 2 && move[0][3] === 2);
    const spatial = moves.filter(move => move[1][0] === 2 && move[1][1] === turn);
    assert(spatial.some(move => move[1][2] === 3 && move[1][3] === 2));
    assert(!spatial.some(move => move[1][2] === 4 && move[1][3] === 2));
    assert(!spatial.some(move => move[1][2] <= 1 && move[1][3] === 2));
    assert(!spatial.some(move => move[1][2] === 2 && move[1][3] >= 3));
    assert(!moves.some(move => move[1][0] === 2 && move[1][1] < turn),
      'a time ray cannot cross the missing preceding board');
  }
});

test('seeded mixed multiverses preserve complete move arrays across all piece types', () => {
  let seed = 0x51dca5e;
  const random = maximum => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return Math.floor(seed / 0x100000000 * maximum);
  };
  for (let fixture = 0; fixture < 48; fixture++) {
    const height = 3 + random(4), width = 3 + random(4);
    const board = Array.from({ length: 3 + random(5) }, (_, line) => {
      if (line && random(5) === 0) return null;
      return Array.from({ length: 1 + random(8) }, () => {
        if (random(7) === 0) return null;
        const squares = empty(height, width);
        for (const [rankIndex, rank] of squares.entries()) for (let file = 0; file < rank.length; file++) {
          if (random(7) !== 0) continue;
          let piece = 1 + random(24);
          // Pawns have already promoted when they reach their last rank.
          if ([1, 2, 15, 16].includes(piece)
              && rankIndex === (piece % 2 ? 0 : height - 1)) piece = 4 - piece % 2;
          rank[file] = (random(2) ? -1 : 1) * piece;
        }
        return squares;
      });
    });
    if (fixture % 2) board[0] = null;
    for (const color of [0, 1]) compare(position(board, color), `seeded fixture ${fixture}, color ${color}`);
  }
});

test('special pawn, brawn and king moves continue to use the library semantics', () => {
  const ep = createPosition({ pgn: '1. e4 / a6 2. e5 / d5' });
  assert(compare(ep, 'en passant').some(move => move.length === 3));
  for (const side of ['w', 'b']) {
    const castling = createPosition({
      pgn: `[Board "Custom"]\n[Size "8x8"]\n[r*3k*2r*/8/8/8/8/8/8/R*3K*2R*:0:1:${side}]`,
    });
    assert.equal(compare(castling, `${side} castling`).filter(move => move.length === 4).length, 2);
  }
  const opening = createPosition();
  assert.equal(compare(opening, 'unmoved pawns').filter(move => {
    const [from, to] = move;
    return opening.board[from[0]][from[1]][from[2]][from[3]] === -2 && to[2] - from[2] === 2;
  }).length, 8);

  for (const color of [0, 1]) for (const type of [1, 8]) for (const sign of [1, -1]) {
    const board = universe(color, 'odd'), turn = 4 + color, forward = color ? -1 : 1;
    const rank = color ? 1 : 3, piece = sign * (type * 2 - color);
    board[2][turn][rank][2] = piece;
    board[2][turn][rank + forward][3] = 1 + color;
    const adjacent = raw.pieceFuncs.timelineMove(2, -forward, false);
    board[adjacent][turn][rank + forward][2] = 1 + color;
    for (const choices of [promotions, [10, 9], [], undefined]) {
      const start = { ...position(board, color), promotions: choices };
      const moves = compare(start, `pawn/brawn ${piece}, promotions ${choices}`);
      if (choices?.length) assert(moves.some(move => move[1].length === 5), 'promotion destination is exercised');
    }
  }
});

test('pawn temporal shortcuts preserve missing timelines, past-only neighbors and brawn captures', () => {
  for (const color of [0, 1]) for (const even of [false, true]) {
    const source = even ? (color ? 1 : 2) : 4, turn = 4 + color;
    const forward = color ? -1 : 1;
    const adjacent = raw.pieceFuncs.timelineMove(source, -forward, even);
    const twoAway = raw.pieceFuncs.timelineMove(source, -2 * forward, even);
    const board = [];
    board[0] = even ? null : [];
    board[source] = Array.from({ length: turn + 1 }, () => empty());
    board[twoAway] = Array.from({ length: turn + 1 }, () => empty());
    board[source][turn][2][2] = -(2 - color);
    const label = `color ${color}, ${even ? 'even' : 'odd'} timelines`;
    for (const missing of [null, undefined]) {
      board[adjacent] = missing;
      const moves = compare(position(board, color), `${label}, missing neighbor`);
      assert(moves.length > 0, 'ordinary spatial moves remain available');
      assert(moves.every(move => move[1][0] === source && move[1][1] === turn),
        'an unmoved pawn cannot double-push across a missing timeline');
    }

    const pastNeighbor = structuredClone(board);
    pastNeighbor[adjacent] = [];
    pastNeighbor[adjacent][turn - 2] = empty();
    pastNeighbor[adjacent][turn - 2][2][2] = 1 + color;
    const temporal = compare(position(pastNeighbor, color), `${label}, past-only neighbor`);
    assert(temporal.some(move => move[1][0] === adjacent && move[1][1] === turn - 2),
      'a pawn still captures into a neighbor that has no board at the current time');
    assert(!temporal.some(move => move[1][0] === twoAway),
      'a temporal double push requires the intermediate board at the current time');

    const brawn = structuredClone(board);
    brawn[source][turn][2][2] = -(16 - color);
    brawn[source][turn - 2][2 + forward][2] = 1 + color;
    const captures = compare(position(brawn, color), `${label}, brawn past capture`);
    assert(captures.some(move => move[1][0] === source && move[1][1] === turn - 2
      && move[1][2] === 2 + forward && move[1][3] === 2),
    'brawns can capture into their own past without an adjacent timeline');
  }
});
