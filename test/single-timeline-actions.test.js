import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createPosition, generateActions, generateActionsAsync, pseudoMoves, positionKey, validateAction,
} from '../src/rules.js';

// An unused trailing timeline leaves every rule and move unchanged while
// selecting the general multiboard traversal instead of its direct path.
const generalPosition = position => ({ ...position, board: [...position.board, null] });
const normalize = candidate => {
  const board = candidate.position.board.slice();
  while (board.length && !board.at(-1)) board.pop();
  return { ...candidate, position: { ...candidate.position, board } };
};
function collect(position, options) {
  let ticks = 0;
  const actions = [...generateActions(position, { ...options, tick() { ticks++; } })].map(normalize);
  return { actions, ticks };
}

const fixtures = () => [
  createPosition(),
  createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' }),
  createPosition({ pgn: '1. e4 / a6 2. e5 / d5' }),
  createPosition({ pgn: '[Board "Custom"]\n[r*3k*2r*/8/8/8/8/8/8/R*3K*2R*:0:1:w]' }),
  createPosition({ pgn: '[Board "Custom"]\n[r*3k*2r*/8/8/8/8/8/8/R*3K*2R*:0:1:b]' }),
  createPosition({ pgn: '[Board "Custom"]\n[7k/P7/8/8/8/8/8/K7:0:1:w]' }),
  createPosition({ pgn: '1. e3 / f6 2. Qe2 / Nc6 3. Qh5' }),
];

test('direct single-timeline actions preserve general traversal order, work and legality', () => {
  for (const position of fixtures()) {
    const before = positionKey(position), reference = generalPosition(position);
    const preferred = collect(reference).actions.at(-1)?.moves ?? [pseudoMoves(position)[0]];
    const hints = [undefined, [], preferred, [...preferred, ...preferred], [[[0, 999, 0, 0], [0, 999, 0, 1]]]];
    for (const tacticalOnly of [false, true]) for (const pruneUnsafe of [false, true]) {
      for (const preferredAction of hints) {
        const options = { tacticalOnly, pruneUnsafe, preferredAction,
          skipOptionalSpatial: true, orderMoves: (_current, moves) => moves.toReversed() };
        const actual = collect(position, options), expected = collect(reference, options);
        assert.deepEqual(actual, expected);
        assert.equal(new Set(actual.actions.map(candidate => positionKey(candidate.position))).size, actual.actions.length);
        for (const action of actual.actions) assert.deepEqual(validateAction(position, action.moves), action.position);
      }
    }
    assert.equal(positionKey(position), before);
  }
});

test('direct traversal deduplicates repeated geometry and promotion outcomes with either unsafe cache mode', () => {
  for (const position of fixtures()) {
    position.promotions.push(...position.promotions);
    const generateMoves = current => { const moves = pseudoMoves(current); return [...moves, ...moves]; };
    for (const tacticalOnly of [false, true]) for (const cacheMoves of [false, true]) {
      for (const cacheUnsafeMoves of [false, true]) for (const pruneUnsafe of [false, true]) {
        const options = { tacticalOnly, cacheMoves, cacheUnsafeMoves, pruneUnsafe, generateMoves };
        assert.deepEqual(collect(position, options), collect(generalPosition(position), options));
      }
    }
  }
});

test('direct async batches retain ordering and close on completion, a witness, or cancellation', async () => {
  const position = createPosition({ pgn: '1. e4 / e5 2. Nf3 / Nc6' });
  for (const firstOnly of [false, true]) for (const tacticalOnly of [false, true]) {
    let opened = 0, closed = 0;
    const options = { firstOnly, tacticalOnly,
      async *orderMoves(current, moves, prefix) {
        opened++;
        assert.deepEqual(prefix, []);
        assert.equal(current, position);
        try {
          yield moves.slice(0, 2);
          yield moves;
          yield moves.slice(2).toReversed();
        } finally { closed++; }
      },
    };
    const actual = [];
    for await (const action of generateActionsAsync(position, options)) actual.push(action);
    const expected = collect(position, { firstOnly, tacticalOnly,
      orderMoves: (_current, moves) => [...moves.slice(0, 2), ...moves, ...moves.slice(2).toReversed()] });
    assert.deepEqual(actual, expected.actions);
    assert.equal(closed, opened);
  }
  let closed = false, ticks = 0;
  await assert.rejects(async () => {
    for await (const _action of generateActionsAsync(position, {
      tick() { if (++ticks === 7) throw new Error('cancelled'); },
      async *orderMoves(_current, moves) {
        try { yield moves.slice(0, 2); yield moves.slice(2); }
        finally { closed = true; }
      },
    })) { /* Resume until the deterministic budget interrupts a batch. */ }
  }, /cancelled/);
  assert(closed);
});
