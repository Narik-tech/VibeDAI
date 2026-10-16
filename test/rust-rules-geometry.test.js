import test from 'node:test';
import assert from 'node:assert/strict';
import { applyMove, canSubmit, inCheck, presentTimelines, pseudoMoves, raw } from '../src/rules.js';
import { nativeRequest, nativeTestOptions } from './rust-native-helper.js';

const promotions = [24, 23, 20, 19, 10, 9, 8, 7, 6, 5, 4, 3];
const empty = (height = 5, width = height) => Array.from({ length: height }, () => Array(width).fill(0));
const position = (board, action = 0) => ({ board, action, promotions });

function compare(start, label) {
  const actual = nativeRequest({ type: 'moves', position: start });
  assert.deepEqual(actual.moves, pseudoMoves(start), `${label}: exact component order`);
  assert.deepEqual(actual.active, raw.boardFuncs.active(start.board), `${label}: active timelines`);
  assert.deepEqual(actual.present, presentTimelines(start), `${label}: present timelines`);
  assert.equal(actual.inCheck, inCheck(start), `${label}: forced-pass royal safety`);
  assert.equal(actual.canSubmit, canSubmit(start), `${label}: submission safety`);
  return actual.moves;
}

function universe(color, layout) {
  const board = Array.from({ length: 7 }, () => Array.from({ length: 7 + color }, () => empty()));
  board[2].length = 5 + color;
  if (layout === 'even') board[0] = null;
  if (layout === 'sparse') {
    board[3] = null;
    board[0][2 + color] = null;
    board[1][4 + color] = null;
    board[2][2 + color] = null;
    board[4].length = 3 + color;
    board[5][6 + color] = null;
  }
  return board;
}

test('Rust geometry retains all piece directions and order across odd, even and sparse timelines', nativeTestOptions, () => {
  for (let type = 1; type <= 12; type++) {
    for (const color of [0, 1]) for (const sign of [1, -1]) {
      for (const layout of ['odd', 'even', 'sparse']) {
        const board = universe(color, layout), turn = 4 + color;
        const piece = sign * (type * 2 - color);
        board[2][turn][2][2] = piece;
        compare(position(board, color), `piece ${piece}, ${layout}`);
      }
    }
  }
});

test('Rust rules agree on seeded fairy-piece multiverses, historical royals and branch applications', nativeTestOptions, () => {
  let seed = 0x51dca5e;
  const random = maximum => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return Math.floor(seed / 0x1_0000_0000 * maximum);
  };
  for (let fixture = 0; fixture < 32; fixture++) {
    const height = 3 + random(4), width = 3 + random(4);
    const board = Array.from({ length: 3 + random(5) }, (_, line) => {
      if (line && random(5) === 0) return null;
      return Array.from({ length: 1 + random(8) }, () => {
        if (random(7) === 0) return null;
        const squares = empty(height, width);
        for (const [rankIndex, rank] of squares.entries()) for (let file = 0; file < rank.length; file++) {
          if (random(7) !== 0) continue;
          let piece = 1 + random(24);
          if ([1, 2, 15, 16].includes(piece) && rankIndex === (piece % 2 ? 0 : height - 1)) piece = 4 - piece % 2;
          rank[file] = (random(2) ? -1 : 1) * piece;
        }
        return squares;
      });
    });
    if (fixture % 2) board[0] = null;
    for (const color of [0, 1]) {
      const start = { ...position(board, color), promotions: [promotions, [], [9], [10]][fixture % 4] };
      const moves = compare(start, `seeded fixture ${fixture}, color ${color}`);
      const temporal = moves.find(move => move[0][0] !== move[1][0] || move[0][1] !== move[1][1]);
      for (const move of [moves[0], temporal].filter(Boolean)) {
        const expected = applyMove(start, move);
        const actual = nativeRequest({ type: 'apply', position: start, action: [move], submit: false });
        assert.deepEqual(actual, JSON.parse(JSON.stringify(expected)), `fixture ${fixture}: immutable component application`);
        compare(expected, `fixture ${fixture}, color ${color}, after component`);
      }
    }
  }
});
