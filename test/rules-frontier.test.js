import test from 'node:test';
import assert from 'node:assert/strict';
import { applyMove, createPosition, parseMove, presentTimelines, pseudoMoves, raw } from '../src/rules.js';

const squares = () => [[12, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
const position = board => ({ board, action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3] });

test('frontier present calculation matches upstream for sparse, inactive and trailing empty histories', () => {
  let seed = 417;
  const random = limit => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed % limit;
  };
  const boards = [[], [[]], [[null]], [[squares(), null, null], null, []]];
  for (let sample = 0; sample < 400; sample++) {
    const board = Array(random(20));
    for (let line = 0; line < board.length; line++) {
      if (random(3) === 0) continue;
      const timeline = board[line] = Array(random(48));
      for (let turn = 0; turn < timeline.length; turn++) {
        if (random(5) === 0) timeline[turn] = squares();
        else if (random(2) === 0) timeline[turn] = null;
      }
    }
    boards.push(board);
  }
  for (const board of boards) for (const action of [0, 1, 4, 5]) {
    assert.deepEqual(presentTimelines({ board, action }), raw.boardFuncs.present(board, action));
  }
});

test('copy-on-write move application matches upstream and preserves every frozen parent', () => {
  const first = squares(); first[1][1] = 4;
  const freeze = value => {
    if (Array.isArray(value)) { for (const item of value) freeze(item); Object.freeze(value); }
    return value;
  };
  const positions = [
    createPosition(),
    createPosition({ pgn: '1. Nf3 / Nf6 2. Nc3 / Nc6' }),
    createPosition({ pgn: '1. e4 / a6 2. e5 / d5' }),
    createPosition({ pgn: '[Board "Custom"]\n[k7/8/8/8/8/8/8/4K*2R*:0:1:w]' }),
    createPosition({ pgn: '[Board "Custom"]\n[7k/1P6/8/8/8/8/8/K7:0:1:w]' }),
    position([[first], [squares()], [squares(), squares(), squares()]]),
  ];
  for (const start of positions) {
    freeze(start.board);
    const before = structuredClone(start);
    for (const move of pseudoMoves(start)) {
      const expected = structuredClone(start);
      raw.boardFuncs.move(expected.board, move);
      assert.deepEqual(applyMove(start, move), expected);
    }
    assert.deepEqual(start, before);
  }
});

test('spatial move ranks stay immutable across siblings, later moves and castling flags', () => {
  const start = createPosition({ pgn: '[Board "Custom"]\n[k7/8/8/8/8/8/8/4K*2R*:0:1:w]' });
  const before = structuredClone(start);
  const moved = applyMove(start, parseMove(start, 'Rh2'));
  const old = start.board[0][0], latest = moved.board[0][1];
  assert.equal(latest[0][7], 0);
  assert.equal(latest[1][7], 8, 'rook movement clears the unmoved flag');
  assert.equal(latest[0][4], -12, 'the unmoved king retains its castling flag');
  assert.notEqual(latest[0], old[0]);
  assert.notEqual(latest[1], old[1]);
  assert.equal(latest[2], old[2], 'untouched ranks share immutable storage');
  const reply = { ...moved, action: 1 };
  const movedBeforeReply = structuredClone(moved);
  applyMove(reply, parseMove(reply, 'Kb8'));
  applyMove(start, parseMove(start, 'O-O'));
  assert.deepEqual(moved, movedBeforeReply);
  assert.deepEqual(start, before);
});

test('move application shares untouched timelines and detaches both sides of a temporal arrival', () => {
  const first = squares(); first[0][1] = 4;
  const start = position([[first], [squares()], [squares()]]);
  const spatial = applyMove(start, parseMove(start, [[0, 0, 0, 0], [0, 0, 1, 0]]));
  assert.notEqual(spatial.board[0], start.board[0]);
  assert.equal(spatial.board[1], start.board[1]);
  assert.equal(spatial.board[2], start.board[2]);
  const arrival = applyMove(start, parseMove(start, [[0, 0, 0, 1], [2, 0, 0, 2]]));
  assert.notEqual(arrival.board[0], start.board[0]);
  assert.notEqual(arrival.board[2], start.board[2]);
  assert.equal(arrival.board[1], start.board[1]);
  assert.equal(start.board[0].length, 1);
  assert.equal(start.board[2].length, 1);
});
