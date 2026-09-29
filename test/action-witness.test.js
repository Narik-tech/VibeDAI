import test from 'node:test';
import assert from 'node:assert/strict';
import { createPosition, generateActions, generateActionsAsync, positionKey, validateAction } from '../src/rules.js';

const squares = () => [[12, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
const promotions = [10, 9, 8, 7, 6, 5, 4, 3];

function probe(position, options, firstOnly) {
  let ticks = 0, keys = 0, skipped = 0;
  const iterator = generateActions(position, {
    ...options, firstOnly,
    tick() { assert(++ticks < 20000); },
    keyPosition(current) { keys++; return positionKey(current); },
    onSkipOptionalSpatial() { skipped++; },
  });
  const first = iterator.next();
  const before = ticks;
  if (firstOnly) {
    assert.equal(iterator.next().done, true, 'a witness iterator must stop after its first result');
    assert.equal(ticks, before, 'closing a witness must not explore more turns');
  } else iterator.return();
  return { first, ticks, keys, skipped };
}

test('existence probes preserve the first complete turn and omit its final history key', () => {
  const multiboard = { board: [[squares()], null, [squares()]], action: 0, promotions };
  const capture = structuredClone(multiboard);
  capture.board[2][0][0][1] = 8;
  capture.board[2][0][1][1] = 9;
  const optionalCapture = structuredClone(capture);
  optionalCapture.board[2].unshift(squares(), squares());
  const temporal = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  const terminal = createPosition({ pgn: '1. e3 / f6 2. Qe2 / Nc6 3. Qh5' });
  const submitted = { ...multiboard, action: 1 };
  for (const position of [createPosition(), multiboard, capture, optionalCapture, temporal, terminal, submitted]) {
    const original = structuredClone(position);
    for (const tacticalOnly of [false, true]) for (const skipOptionalSpatial of [false, true]) {
      const options = { tacticalOnly, skipOptionalSpatial, orderMoves: (_, moves) => moves.toReversed() };
      const normal = probe(position, options, false);
      const witness = probe(position, options, true);
      assert.deepEqual(witness.first, normal.first);
      assert.equal(witness.ticks, normal.ticks);
      assert.equal(witness.skipped, normal.skipped);
      if (!normal.first.done) {
        assert(witness.keys <= normal.keys);
        if (!tacticalOnly && position.board.length === 1 && witness.first.value.moves.length) assert(witness.keys < normal.keys);
        validateAction(position, witness.first.value.moves);
        const preferred = { ...options, preferredAction: normal.first.value.moves };
        const replay = probe(position, preferred, true);
        assert.deepEqual(replay.first, probe(position, preferred, false).first);
        assert.equal(replay.keys, 0, 'a valid preferred witness needs no deduplication keys');
        const stale = { ...options, preferredAction: [[[0, 99, 0, 0], [0, 99, 0, 1]]] };
        assert.deepEqual(probe(position, stale, true).first, probe(position, stale, false).first);
      }
    }
    assert.deepEqual(position, original);
  }
});

test('async existence probes stop after the same first legal submission', async () => {
  const position = { board: [[squares()], null, [squares()]], action: 0, promotions };
  const first = generateActions(position).next().value;
  for (const preferredAction of [undefined, first.moves]) {
    const options = { firstOnly: true, preferredAction };
    const actual = [];
    for await (const candidate of generateActionsAsync(position, options)) actual.push(candidate);
    assert.deepEqual(actual, [...generateActions(position, options)]);
    assert.equal(actual.length, 1);
    validateAction(position, actual[0].moves);
  }
  let opened = 0, closed = 0;
  const candidates = [];
  for await (const candidate of generateActionsAsync(position, {
    firstOnly: true,
    async *orderMoves(_, moves) {
      opened++;
      try {
        yield moves.slice(0, 1);
        yield moves.slice(1);
      } finally { closed++; }
    },
  })) candidates.push(candidate);
  assert.equal(candidates.length, 1);
  assert(opened >= 2, 'the fixture must suspend ordering on multiple boards');
  assert.equal(closed, opened, 'all suspended ordering generators must close');
});

test('empty timelines and trailing gaps still require the full present test', () => {
  const histories = [[[squares(), null]], [[squares(), ,]], [[], null, [squares()]]];
  for (const board of histories) for (const action of [0, 1]) {
    const position = { board, action, promotions };
    for (const skipOptionalSpatial of [false, true]) {
      const options = { skipOptionalSpatial };
      assert.deepEqual(probe(position, options, true).first, probe(position, options, false).first);
    }
  }
});

test('direct single-timeline probes respect unsafe-move caching and close async batches', async () => {
  const position = createPosition({ pgn: '1. e3 / f6 2. Qe2 / Nc6 3. Qh5' });
  for (const pruneUnsafe of [false, true]) for (const cacheUnsafeMoves of [false, true]) {
    const options = { pruneUnsafe, cacheUnsafeMoves };
    const normal = probe(position, options, false), witness = probe(position, options, true);
    assert.deepEqual(witness.first, normal.first);
    assert.equal(witness.ticks, normal.ticks);
    assert.equal(witness.first.done, true);
  }
  let closed = false;
  const initial = createPosition();
  const iterator = generateActionsAsync(initial, {
    firstOnly: true,
    async *orderMoves(_, moves) {
      try {
        yield moves.slice(0, 1);
        yield moves.slice(1);
      } finally { closed = true; }
    },
  });
  const first = await iterator.next();
  validateAction(initial, first.value.moves);
  assert.equal((await iterator.next()).done, true);
  assert(closed);
});
