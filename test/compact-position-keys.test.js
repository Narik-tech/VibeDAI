import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createPosition, createPositionKeyCache, positionKey, generateActions,
  applyMove, parseMove, validateAction,
} from '../src/rules.js';

const positionWithBoard = squares => ({ board: [[squares]], action: 0, promotions: [24, 23, 10, 9] });

function assertSameKeyEquivalence(positions) {
  const compact = createPositionKeyCache({ compact: true });
  const plain = createPositionKeyCache();
  const publicByCompact = new Map(), compactByPublic = new Map();
  for (const position of positions) {
    const publicKey = positionKey(position), compactKey = compact(position);
    assert.equal(plain(position), publicKey, 'the default cache preserves the public key format');
    assert.equal(compact(structuredClone(position)), compactKey, 'equivalent clones have the same compact key');
    assert.equal(compact(position), compactKey, 'repeated lookups retain the same key');
    if (publicByCompact.has(compactKey)) {
      assert.equal(publicByCompact.get(compactKey), publicKey, 'distinct positions must not share compact keys');
    }
    if (compactByPublic.has(publicKey)) {
      assert.equal(compactByPublic.get(publicKey), compactKey, 'equivalent positions share compact keys');
    }
    publicByCompact.set(compactKey, publicKey);
    compactByPublic.set(publicKey, compactKey);
  }
}

test('compact history keys distinguish signed piece flags, row boundaries and fallback values', () => {
  const pieces = Array.from({ length: 65 }, (_, index) => index - 32);
  const boards = [
    ...pieces.map(piece => [[piece]]),
    [[-0]], [], [[]], [[], []], [[1, 2]], [[1], [2]], [[1, 2], []], [[], [1, 2]],
    [[-12, 0, 20], [19, -1, 11, 23, -24]],
    [[33]], [[-33]], [[1.5]], [[null]], [[undefined]], [new Array(1)],
    [[NaN]], [[Infinity]], [[-Infinity]], [null], [undefined], new Array(1),
    [['c[[]]']], [['j[[null]]']], [['[],;:']], [[{ piece: 1 }]],
    [Array(4097).fill(0)],
  ];
  assertSameKeyEquivalence(boards.map(positionWithBoard));
});

test('compact keys preserve sparse timeline coordinates and every historical board', () => {
  const squares = [[-12, 0, 20], [19, -1, 11]];
  const board = [];
  board[2] = [];
  board[2][4] = squares;
  board[5] = [null, structuredClone(squares)];
  const sparse = { board, action: 3, promotions: [24, 23, 10, 9] };
  const denseNulls = JSON.parse(JSON.stringify(sparse));
  const movedTimeline = structuredClone(sparse);
  movedTimeline.board[1] = movedTimeline.board[2];
  movedTimeline.board[2] = null;
  const movedTurn = structuredClone(sparse);
  movedTurn.board[2][3] = movedTurn.board[2][4];
  movedTurn.board[2][4] = null;
  const changedPast = structuredClone(sparse);
  changedPast.board[2][4][0][0] = 12;
  const extraMissingTurn = structuredClone(sparse);
  extraMissingTurn.board[2].length++;
  assertSameKeyEquivalence([sparse, denseNulls, movedTimeline, movedTurn, changedPast, extraMissingTurn]);
});

test('compact keys retain mover and promotion differences while reusing immutable history', () => {
  const position = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  const variants = [];
  for (const action of [0, 1, 4, 5]) {
    for (const promotions of [position.promotions, [10, 9], [9, 10], [], undefined, null]) {
      variants.push({ ...position, action, promotions });
    }
  }
  assertSameKeyEquivalence(variants);
  const compact = createPositionKeyCache({ compact: true });
  const before = compact(position);
  position.promotions.push(24, 23);
  assert.notEqual(compact(position), before, 'mutable promotion lists are encoded on every lookup');
  assert.equal(compact(position), compact(structuredClone(position)));
});

test('compact keys distinguish sibling moves and edits observed by a new search cache', () => {
  const position = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  const siblings = ['Bb5', 'Bc4', 'd4'].map(notation => applyMove(position, parseMove(position, notation)));
  assertSameKeyEquivalence([position, ...siblings]);
  const before = createPositionKeyCache({ compact: true })(position);
  position.board[0][0][1][0] = 0;
  assert.notEqual(createPositionKeyCache({ compact: true })(position), before);
});

test('compact action keys preserve complete actions, preferred replay and work ticks', () => {
  const squares = () => [[12, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
  const multiboard = {
    board: [[squares()], null, [squares(), squares(), squares()]],
    action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3],
  };
  const capture = structuredClone(multiboard);
  capture.board[2][2][0][1] = 4;
  capture.board[2][2][1][2] = 1;
  for (const start of [createPosition(), multiboard, capture]) {
    const preferredAction = [...generateActions(start)].at(-1).moves;
    for (const tacticalOnly of [false, true]) {
      const options = { preferredAction, tacticalOnly };
      let plainTicks = 0, compactTicks = 0;
      const plain = [...generateActions(start, {
        ...options, keyPosition: createPositionKeyCache(), tick: () => plainTicks++,
      })];
      const compact = [...generateActions(start, {
        ...options, keyPosition: createPositionKeyCache({ compact: true }), tick: () => compactTicks++,
      })];
      assert.deepEqual(compact, plain);
      assert.equal(compactTicks, plainTicks);
      for (const action of compact) {
        assert.equal(positionKey(validateAction(start, action.moves)), positionKey(action.position));
      }
    }
  }
});
