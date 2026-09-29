import test from 'node:test';
import assert from 'node:assert/strict';
import { raw, createPosition, pseudoMoves, applyMove, positionKey } from '../src/rules.js';
import { createRoyalSafety } from '../src/royal-safety.js';

const { attackedByNextPlayer, findRoyalAttack } = createRoyalSafety(raw);
const cached = createRoyalSafety(raw, { cacheBoards: true });
const promotions = [10, 9, 8, 7, 6, 5, 4, 3];
const empty = (height = 5, width = height) => Array.from({ length: height }, () => Array(width).fill(0));
function compare(position, label = '') {
  const captures = raw.boardFuncs.moves(position.board, position.action + 1, false, false, false, position.promotions)
    .filter(move => {
      const [l, t, r, f] = move[1], piece = Math.abs(position.board[l]?.[t]?.[r]?.[f]);
      return [11, 12, 19, 20].includes(piece) && piece % 2 === position.action % 2;
    });
  const before = positionKey(position);
  assert.equal(attackedByNextPlayer(position), captures.length > 0, label);
  // A fresh cache sees the same immutable snapshot; repeated calls then reuse
  // its board scans. Mutation coverage below keeps the public default uncached.
  const localCache = createRoyalSafety(raw, { cacheBoards: true });
  assert.equal(localCache.attackedByNextPlayer(position), captures.length > 0, label);
  assert.equal(localCache.attackedByNextPlayer(position), captures.length > 0, label);
  const witness = findRoyalAttack(position);
  assert.equal(Boolean(witness), captures.length > 0, label);
  if (witness) assert(captures.some(move => JSON.stringify(move) === JSON.stringify(witness)), `${label}: exact capture witness`);
  assert.equal(positionKey(position), before);
}

test('direct royal attacks agree for all piece types, both colors, and odd/even timelines', () => {
  for (const color of [0, 1]) for (const even of [false, true]) for (let type = 1; type <= 12; type++) {
    const board = [];
    for (let l = even ? 1 : 0; l < 7; l++) board[l] = Array.from({ length: 7 + color }, () => empty());
    const source = [2, 6 + color, 2, 2], piece = type * 2 - color;
    board[source[0]][source[1]][source[2]][source[3]] = -piece;
    const generated = raw.pieceFuncs.moves(board, source, false, promotions);
    for (const move of generated) {
      const [l, t, r, f] = move[1];
      for (const royal of [color ? 12 : 11, color ? 20 : 19]) {
        board[l][t][r][f] = royal;
        compare({ board, action: 1 - color, promotions }, `${piece} to ${move[1]}`);
        board[l][t][r][f] = 0;
      }
    }
  }
});

test('promotion rules distinguish spatial and extra brawn captures from pawn time captures', () => {
  for (const color of [0, 1]) for (const type of [1, 8]) {
    const forward = color ? -1 : 1, rank = color ? 1 : 3, piece = type * 2 - color;
    const board = Array.from({ length: 5 }, () => Array.from({ length: 5 + color }, () => empty()));
    const l = 2, t = 4 + color, royal = color ? 12 : 11;
    board[l][t][rank][2] = piece;
    const position = { board, action: 1 - color };
    const targets = [[l, t, rank + forward, 3],
      [raw.pieceFuncs.timelineMove(l, -forward, false), t, rank + forward, 2],
      [l, t - 2, rank + forward, 2],
      [raw.pieceFuncs.timelineMove(l, -forward, false), t - 2, rank, 2]];
    for (const [tl, tt, tr, tf] of targets) {
      board[tl][tt][tr][tf] = royal;
      for (const choices of [undefined, [], [color ? 10 : 9], [color ? 9 : 10], promotions]) {
        position.promotions = choices;
        compare(position, `promotion ${piece} ${choices}`);
      }
      board[tl][tt][tr][tf] = 0;
    }
    // Empty promotion choices fall back to types found anywhere in history.
    board[0][0][0][0] = 6;
    board[l][t][rank + forward][3] = royal;
    position.promotions = [];
    compare(position);
    assert(attackedByNextPlayer(position));
  }
});

test('historical blockers and missing boards stop rays while knight jumps cross gaps', () => {
  const board = [Array.from({ length: 8 }, () => empty())];
  const position = { board, action: 0, promotions };
  board[0][7][2][2] = 7;
  board[0][1][2][2] = 20;
  compare(position);
  assert(attackedByNextPlayer(position));
  board[0][3][2][2] = 2;
  compare(position);
  assert.equal(attackedByNextPlayer(position), false);
  board[0][3] = null;
  compare(position);
  assert.equal(attackedByNextPlayer(position), false);
  // A knight reaches two full turns back without visiting the absent board.
  board[0][7][2][2] = 5;
  board[0][3] = empty();
  board[0][3][3][2] = 12;
  board[0][5] = null;
  compare(position);
  assert(attackedByNextPlayer(position));
  // The same public checker must see edits to a previously inspected snapshot.
  board[0][3][3][2] = 0;
  compare(position);
  assert.equal(attackedByNextPlayer(position), false);
});

test('seeded sparse histories match full move enumeration and detect mutations between calls', () => {
  let seed = 0x19cafe;
  const random = n => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return Math.floor(seed / 0x1_0000_0000 * n); };
  for (let sample = 0; sample < 180; sample++) {
    const height = 3 + random(5), width = 3 + random(5), board = [];
    const even = random(2), action = random(2);
    for (let l = even ? 1 : 0; l < 7; l++) {
      if (l > 2 && !random(4)) continue;
      const timeline = board[l] = [];
      const last = 1 + random(8);
      for (let t = 0; t <= last; t++) {
        if (t < last && !random(5)) { timeline[t] = null; continue; }
        const squares = timeline[t] = empty(height, width);
        for (let count = 0; count < 5; count++) {
          const type = 1 + random(12), color = random(2), r = 1 + random(height - 2), f = random(width);
          squares[r][f] = (random(2) ? -1 : 1) * (type * 2 - color);
        }
        if (!random(3)) squares[random(height)][random(width)] = (random(2) ? 12 : 20) - action;
      }
    }
    const position = { board, action, promotions: [promotions, [], [9], [10]][random(4)] };
    compare(position, `seeded sample ${sample}`);
    for (const timeline of board) for (const squares of timeline || []) for (const rank of squares || []) {
      for (let f = 0; f < rank.length; f++) if ([11, 12, 19, 20].includes(Math.abs(rank[f]))) rank[f] = 0;
    }
    compare(position, `mutated sample ${sample}`);
    assert.equal(attackedByNextPlayer(position), false);
  }
});

test('real histories and every next component retain upstream attack results', () => {
  for (const setup of [{}, { variant: 'two_timelines' }, { pgn: '1. Nf3 / Nf6 2. Nc3 / Nc6' },
    { pgn: '1. e4 / a6 2. e5 / d5' }, { pgn: '[Board "Custom"]\n[r*3k*2r*/8/8/8/8/8/8/R*3K*2R*:0:1:w]' }]) {
    const start = createPosition(setup);
    compare(start);
    for (const move of pseudoMoves(start)) {
      const next = applyMove(start, move);
      compare(next);
      assert.equal(cached.attackedByNextPlayer(next), attackedByNextPlayer(next));
      compare({ ...next, action: next.action + 1 });
    }
  }
});
