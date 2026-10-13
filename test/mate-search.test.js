import test from 'node:test';
import assert from 'node:assert/strict';
import { scanMateInOne } from '../src/mate-search.js';
import {
  createPosition, createPositionKeyCache, createSearchMoveGenerator, createSearchRoyalSafety,
  formatAction, generateActions, inCheck, positionKey, validateAction,
} from '../src/rules.js';

const promotions = [10, 9, 8, 7, 6, 5, 4, 3];
const unlimited = { maxWork: Infinity, timeMs: Infinity };
const smallBoard = () => [[12, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
const matingBoard = () => [[0, 10, 0, 0], [0, 0, 12, 0], [0, 0, 0, 0], [0, 0, 0, 11]];

function assertMate(position, result) {
  assert.equal(result.status, 'mate');
  assert.equal(result.reason, null);
  const next = validateAction(position, result.action);
  assert(inCheck(next));
  assert.equal(generateActions(next, { firstOnly: true }).next().done, true);
}

test('proves temporal mates for either mover without changing the position', () => {
  for (const [pgn, notation] of [
    ['1. e3 / f6 2. Qe2 / Nc6', /Qh5/],
    ['1. f3 / e6 2. Nc3 / Qe7 3. a3', /Qh4/],
  ]) {
    const position = createPosition({ pgn }), original = positionKey(position);
    let charged = 0;
    const result = scanMateInOne(position, {
      ...unlimited, tick: () => charged++,
      keyPosition: createPositionKeyCache({ compact: true }),
      generateMoves: createSearchMoveGenerator(), royalSafety: createSearchRoyalSafety(),
    });
    assertMate(position, result);
    assert.match(formatAction(position, result.action), notation);
    assert.equal(charged, result.work);
    assert.equal(positionKey(position), original);
    assert.equal(Object.hasOwn(result, 'position'), false);
  }
});

test('proves a compound mate only after completing both required boards', () => {
  const position = { board: [[matingBoard()], null, [matingBoard()]], action: 0, promotions };
  const result = scanMateInOne(position, unlimited);
  assertMate(position, result);
  assert.equal(result.action.length, 2);
  assert.equal(new Set(result.action.map(move => move[0][0])).size, 2);
});

test('exhaustion includes optional component moves after valid submissions', () => {
  const position = {
    board: [[smallBoard()], null, [matingBoard(), matingBoard(), matingBoard()]],
    action: 0, promotions,
  };
  const actions = [...generateActions(position)];
  const restricted = [...generateActions(position, { skipOptionalSpatial: true })];
  assert(actions.some(candidate => candidate.moves.length === 1));
  assert(actions.some(candidate => candidate.moves.length === 2));
  assert(actions.length > restricted.length);
  const result = scanMateInOne(position, unlimited);
  assert.equal(result.status, 'no-mate');
  assert.equal(result.candidates, actions.length);
  assert.equal(result.action, null);
  assert.equal(result.reason, null);
});

test('an empty present-board stage still explores legal optional-board escapes', () => {
  const future = [[0, 8, 0], [11, 0, 0], [4, 0, 12]];
  const position = {
    board: [[[[0, 11, 12], [7, 0, 0], [3, 0, 0]]], null,
      Array.from({ length: 3 }, () => structuredClone(future))],
    action: 0, promotions,
  };
  assert(inCheck(position));
  assert.equal(generateActions(position, { skipOptionalSpatial: true }).next().done, true);
  validateAction(position, [[[2, 2, 0, 1], [2, 2, 0, 2]], [[0, 0, 0, 2], [2, 2, 0, 2]]]);
  const staged = scanMateInOne(position, unlimited);
  const exhaustive = scanMateInOne(position, { ...unlimited, preferPresent: false });
  assert.equal(staged.status, exhaustive.status);
  assert.equal(staged.candidates, exhaustive.candidates);
  assert(staged.candidates > 0);
});

test('checks with legal escapes and stalemate do not count as mate', () => {
  const board = smallBoard(); board[1][1] = 8;
  const escape = scanMateInOne({ board: [[board]], action: 0, promotions }, unlimited);
  assert.equal(escape.status, 'no-mate');
  assert(escape.checks > 0);
  const stalemate = { board: [[[[12, 0, 0], [0, 0, 11], [0, 9, 0]]]], action: 0, promotions };
  assert.equal(inCheck(stalemate), false);
  const result = scanMateInOne(stalemate, unlimited);
  assert.equal(result.status, 'no-mate');
  assert.equal(result.candidates, 0);
  assert.equal(result.action, null);
});

test('local work limits remain deterministic and cannot disprove interrupted mate', () => {
  const position = createPosition({ pgn: '1. e3 / f6 2. Qe2 / Nc6' });
  const complete = scanMateInOne(position, unlimited);
  assertMate(position, complete);
  for (const maxWork of [0, 1, 10, complete.work - 1]) {
    let charged = 0;
    const options = { maxWork, timeMs: Infinity };
    const first = scanMateInOne(position, { ...options, tick: () => charged++ });
    const second = scanMateInOne(position, options);
    assert.deepEqual(first, second);
    assert.equal(first.status, 'unknown');
    assert.equal(first.reason, 'work');
    assert.equal(first.work, maxWork);
    assert.equal(charged, maxWork);
    assert.equal(first.action, null);
    if (maxWork === complete.work - 1) assert(first.checks > 0);
  }
  assertMate(position, scanMateInOne(position, { maxWork: complete.work, timeMs: Infinity }));
});

test('local timeout returns unknown while caller cancellation propagates unchanged', () => {
  const position = createPosition(), original = positionKey(position);
  const timed = scanMateInOne(position, { maxWork: Infinity, timeMs: 0 });
  assert.equal(timed.status, 'unknown');
  assert.equal(timed.reason, 'time');
  assert.equal(timed.work, 0);
  const cancelled = new Error('cancelled by caller');
  let work = 0;
  assert.throws(() => scanMateInOne(position, {
    ...unlimited, tick() { if (++work === 7) throw cancelled; },
  }), error => error === cancelled);
  assert.equal(positionKey(position), original);
});

test('preferred actions and exhaustive component ordering do not change proof semantics', () => {
  const position = createPosition({ pgn: '1. e3 / f6 2. Qe2 / Nc6' });
  const normal = scanMateInOne(position, unlimited);
  let ordered = 0;
  const preferred = scanMateInOne(position, {
    ...unlimited, preferredAction: normal.action,
    orderMoves: (_position, moves) => { ordered++; return moves.toReversed(); },
  });
  assertMate(position, preferred);
  assert.deepEqual(preferred.action, normal.action);
  assert.equal(preferred.candidates, 1);
  assert(ordered > 0);
});
