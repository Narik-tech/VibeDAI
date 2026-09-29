import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyMove, createPositionKeyCache, generateActions, parseMove, positionKey, validateAction,
} from '../src/rules.js';

const squares = () => [[12, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
const position = board => ({ board, action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3] });
const history = (board, length = 5) => Array.from({ length }, () => structuredClone(board));

function traversal(start, keyPosition, options = {}) {
  let ticks = 0, skipped = 0;
  const states = [];
  const actions = [...generateActions(start, {
    ...options, keyPosition,
    tick() { assert(++ticks < 10000, 'miniature fixture exceeded its traversal budget'); },
    onSkipOptionalSpatial() { skipped++; },
    orderMoves(current, moves) {
      states.push(current);
      return moves.toReversed();
    },
  })];
  return { actions, ticks, skipped, states };
}

function assertEquivalent(start, options = {}) {
  const original = structuredClone(start);
  const plain = traversal(start, positionKey, options);
  const key = createPositionKeyCache();
  const cached = traversal(start, key, options);
  assert.deepEqual(cached, plain, 'action keys must preserve traversal order, work and policy evidence');

  // Check both directions: unequal full histories must never collide, and
  // equivalent histories reached by different component orders must dedupe.
  const actionKey = key.forAction(start), compactToFull = new Map(), fullToCompact = new Map();
  for (const current of [start, ...cached.states, ...cached.actions.map(action => ({
    ...action.position, action: start.action,
  }))]) {
    const compact = actionKey(current), full = positionKey(current);
    if (compactToFull.has(compact)) assert.equal(compactToFull.get(compact), full);
    if (fullToCompact.has(full)) assert.equal(fullToCompact.get(full), compact);
    compactToFull.set(compact, full);
    fullToCompact.set(full, compact);
    assert.equal(key(current), full, 'the full search-table key remains exact');
  }
  for (const action of cached.actions) {
    assert.equal(positionKey(validateAction(start, action.moves)), positionKey(action.position));
  }
  assert.deepEqual(start, original);
  return cached;
}

test('preferred empty turns are emitted once with cached action keys', () => {
  const start = position([[squares(), squares()]]);
  for (const preferredAction of [undefined, []]) {
    const result = assertEquivalent(start, { preferredAction });
    assert.equal(result.actions.length, 1);
    assert.deepEqual(result.actions[0].moves, []);
  }
  assert.deepEqual(assertEquivalent(start, { preferredAction: [], tacticalOnly: true }).actions, []);
});

test('independent multiboard move orders share the same exact action key', () => {
  const start = position([[squares()], null, [squares()]]);
  const left = parseMove(start, [[0, 0, 0, 0], [0, 0, 0, 1]]);
  const right = parseMove(start, [[2, 0, 0, 0], [2, 0, 0, 1]]);
  const leftFirst = applyMove(applyMove(start, left), right);
  const rightFirst = applyMove(applyMove(start, right), left);
  const actionKey = createPositionKeyCache().forAction(start);
  assert.equal(positionKey(leftFirst), positionKey(rightFirst));
  assert.equal(actionKey(leftFirst), actionKey(rightFirst));
  assert.notEqual(actionKey(applyMove(start, left)), actionKey(applyMove(start, right)));
  const result = assertEquivalent(start, { preferredAction: [right, left] });
  assert.deepEqual(result.actions[0].moves, [right, left]);
  assert.equal(result.actions.filter(action =>
    positionKey({ ...action.position, action: start.action }) === positionKey(leftFirst)).length, 1);
});

test('temporal branches retain their destination time and assigned timeline indices', () => {
  const bishop = squares(); bishop[1][2] = 4;
  const start = position([history(bishop), history(squares())]);
  const first = parseMove(start, [[0, 4, 1, 2], [0, 2, 1, 1]]);
  const second = parseMove(start, [[1, 4, 0, 0], [1, 2, 0, 1]]);
  const firstThenSecond = applyMove(applyMove(start, first), second);
  const secondThenFirst = applyMove(applyMove(start, second), first);
  const actionKey = createPositionKeyCache().forAction(start);
  assert(firstThenSecond.board[2] && firstThenSecond.board[4]);
  assert(secondThenFirst.board[2] && secondThenFirst.board[4]);
  assert.notEqual(positionKey(firstThenSecond), positionKey(secondThenFirst));
  assert.notEqual(actionKey(firstThenSecond), actionKey(secondThenFirst),
    'swapping which branch owns each timeline must remain distinct');

  const earlier = applyMove(start, parseMove(start, [[0, 4, 1, 2], [0, 0, 1, 0]]));
  const later = applyMove(start, first);
  assert.equal(earlier.board[2].length, 2);
  assert.equal(later.board[2].length, 4);
  assert.notEqual(actionKey(earlier), actionKey(later));
});

test('sparse histories and newly allocated branches preserve complete traversal and work counts', () => {
  const bishop = squares(); bishop[1][2] = 4;
  const sparse = [];
  sparse[0] = [];
  sparse[0][4] = bishop;
  sparse[3] = [];
  sparse[3][0] = squares();
  const captureBranch = position([history(bishop), [squares()]]);
  captureBranch.board[0][2][1][1] = 1;
  const starts = [position(sparse), captureBranch];
  for (const start of starts) {
    const preferredAction = [...generateActions(start)].at(-1).moves;
    for (const skipOptionalSpatial of [false, true]) for (const tacticalOnly of [false, true]) {
      const result = assertEquivalent(start, { preferredAction, skipOptionalSpatial, tacticalOnly });
      if (!tacticalOnly) assert(result.actions.length > 0);
      if (tacticalOnly && start === captureBranch) assert(result.actions.length > 0);
    }
  }
});

test('duplicate promotion choices deduplicate identically with full and action keys', () => {
  const board = squares(); board[2][1] = 2;
  const start = { ...position([[board]]), promotions: [10, 9, 10, 9, 4, 3] };
  const before = structuredClone(start.promotions);
  for (const tacticalOnly of [false, true]) {
    const result = assertEquivalent(start, { tacticalOnly });
    const promotions = result.actions.filter(action => action.moves.some(move => move[1].length > 4));
    assert.equal(promotions.length, 2, 'duplicate queen choices must not produce duplicate submissions');
    assert.equal(new Set(result.actions.map(action => positionKey(action.position))).size, result.actions.length);
  }
  assert.deepEqual(start.promotions, before);
});
