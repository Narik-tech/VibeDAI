import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createPosition, createPositionKeyCache, positionKey, generateActions,
  applyMove, parseMove, validateAction,
} from '../src/rules.js';

test('search key cache preserves exact JSON history, sparse coordinates and piece flags', () => {
  const key = createPositionKeyCache();
  const squares = [[-12, 0, 20], [19, -1, 11]];
  const board = [];
  board[2] = [];
  board[2][4] = squares;
  board[5] = [null, structuredClone(squares)];
  const sparse = { board, action: 3, promotions: [24, 23, 10, 9] };
  const start = createPosition({ pgn: '1. Nf3 / Nf6 2. Nc3 / Nc6' });
  const branch = validateAction(start, [parseMove(start, '(0T3)Nc3>>(0T2)c5')]);
  for (const position of [sparse, start, branch, { ...branch, action: 6 }, { ...branch, promotions: [10, 9] }]) {
    assert.equal(key(position), positionKey(position));
    assert.equal(key(position), positionKey(position), 'repeated lookups remain byte-for-byte equivalent');
  }
  const changedPast = structuredClone(branch);
  changedPast.board[0][0][1][0] = 0;
  assert.notEqual(key(branch), key(changedPast));
  assert.equal(key(changedPast), positionKey(changedPast));
});

test('public keys and newly scoped search caches observe edits to previously keyed positions', () => {
  const position = createPosition();
  const firstCache = createPositionKeyCache();
  const before = firstCache(position);
  assert.equal(before, positionKey(position));
  position.board[0][0][1][0] = 0;
  assert.notEqual(positionKey(position), before);
  assert.equal(createPositionKeyCache()(position), positionKey(position));
  position.promotions.push(24, 23);
  assert.equal(createPositionKeyCache()(position), positionKey(position));
});

test('cached action keys preserve complete actions, preferred replay and work ticks', () => {
  const square = () => [[12, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
  const multiboard = {
    board: [[square()], null, [square(), square(), square()]],
    action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3],
  };
  const capture = structuredClone(multiboard);
  capture.board[2][2][0][1] = 4;
  capture.board[2][2][1][2] = 1;
  for (const start of [createPosition(), multiboard, capture]) {
    const preferredAction = [...generateActions(start)].at(-1).moves;
    for (const tacticalOnly of [false, true]) {
      const options = { preferredAction, tacticalOnly };
      let plainTicks = 0, cachedTicks = 0;
      const plain = [...generateActions(start, { ...options, tick: () => plainTicks++ })];
      const cached = [...generateActions(start, { ...options, keyPosition: createPositionKeyCache(), tick: () => cachedTicks++ })];
      assert.deepEqual(cached, plain);
      assert.equal(cachedTicks, plainTicks);
      for (const action of cached) assert.equal(positionKey(validateAction(start, action.moves)), positionKey(action.position));
    }
  }
});

test('shared immutable histories remain distinct after sibling moves and submissions', () => {
  const position = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  const cache = createPositionKeyCache();
  const before = cache(position);
  for (const notation of ['Bb5', 'Bc4', 'd4']) {
    const child = applyMove(position, parseMove(position, notation));
    assert.equal(cache(child), positionKey(child));
    assert.equal(cache({ ...child, action: child.action + 1 }), positionKey({ ...child, action: child.action + 1 }));
    assert.notEqual(cache(child), before);
    assert.equal(cache(position), before);
  }
});

test('a reused immutable history retains independent mover and promotion keys', () => {
  const position = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  const cache = createPositionKeyCache();
  const keys = new Set();
  for (const action of [4, 5]) for (const promotions of [position.promotions, [10, 9], [], undefined]) {
    const variant = { ...position, action, promotions };
    const key = cache(variant);
    assert.equal(key, positionKey(variant));
    assert.equal(cache(variant), key);
    keys.add(key);
  }
  assert.equal(keys.size, 8);
  const before = cache(position);
  position.promotions.push(24, 23);
  assert.notEqual(cache(position), before);
  assert.equal(cache(position), positionKey(position));
});
