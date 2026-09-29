import test from 'node:test';
import assert from 'node:assert/strict';
import {
  raw, generateActions, applyMove, parseMove, canSubmit, inCheck, positionKey, validateAction,
  createPosition, pseudoMoves, isTacticalMove,
} from '../src/rules.js';

const clearBoard = () => [[12, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 11]];
const checkedBoard = () => { const board = clearBoard(); board[3][0] = 7; return board; };
const position = board => ({ board, action: 0, promotions: [10, 9, 8, 7, 6, 5, 4, 3] });

function collect(start, pruneUnsafe) {
  let ticks = 0;
  const actions = [...generateActions(start, { pruneUnsafe, tick() { ticks++; } })];
  return { actions, ticks, keys: actions.map(action => positionKey(action.position)).sort() };
}

test('actual opponent threat prunes a dead partial turn without exploring optional moves', () => {
  // Timeline 0 has already advanced to Black, whose rook can capture Ka1.
  // White still has a playable board on timeline +1, but cannot change the
  // existing Black board or its already available royal-capture move.
  const start = position([[checkedBoard(), checkedBoard()], null, [clearBoard()]]);
  const baseline = collect(start, false), pruned = collect(start, true);
  assert.deepEqual(baseline.keys, []);
  assert.deepEqual(pruned.keys, baseline.keys);
  assert(pruned.ticks < baseline.ticks, `${pruned.ticks} should improve on ${baseline.ticks}`);
});

test('safe pruning preserves every legal complete action on two and three active boards', () => {
  for (const board of [
    [[checkedBoard()], null, [checkedBoard()]],
    [[checkedBoard()], [checkedBoard()], [checkedBoard()]],
  ]) {
    const start = position(board), before = positionKey(start);
    const baseline = collect(start, false), pruned = collect(start, true);
    assert(baseline.actions.length > 0);
    assert.deepEqual(pruned.keys, baseline.keys);
    assert(pruned.ticks < baseline.ticks, `${board.length} containers: ${pruned.ticks} vs ${baseline.ticks}`);
    assert.equal(positionKey(start), before);
    for (const action of pruned.actions) assert.equal(positionKey(validateAction(start, action.moves)), positionKey(action.position));
  }
});

test('learning unsafe spatial outcomes saves work without losing multiboard evasions', () => {
  const start = position([[checkedBoard()], [checkedBoard()], [checkedBoard()]]);
  const before = positionKey(start);
  const run = options => {
    let ticks = 0;
    const actions = [...generateActions(start, { ...options, tick() { ticks++; } })];
    return { ticks, keys: actions.map(action => positionKey(action.position)).sort() };
  };
  const exhaustive = run({ pruneUnsafe: false, cacheMoves: false });
  const unlearned = run({ cacheUnsafeMoves: false });
  for (const cacheMoves of [true, false]) {
    const learned = run({ cacheMoves });
    assert.deepEqual(learned.keys, exhaustive.keys);
    assert.deepEqual(learned.keys, unlearned.keys);
    assert(learned.ticks < unlearned.ticks, `${learned.ticks} vs ${unlearned.ticks}`);
  }
  assert.equal(positionKey(start), before);
});

test('a spatial move unsafe with one companion remains available with a different companion', () => {
  const empty = () => Array.from({ length: 5 }, () => Array(5).fill(0));
  const left = empty(), right = empty();
  left[0][0] = 12; left[0][2] = 8; left[2][2] = 7; left[4][4] = 11;
  right[1][2] = 12; right[4][4] = 11;
  // Advancing both boards lets the Black rook capture across timelines. The
  // same king move is safe when its companion captures that rook instead.
  const start = position([[left], null, [right]]);
  const advance = parseMove(start, [[0, 0, 0, 0], [0, 0, 0, 1]]);
  const king = parseMove(start, [[2, 0, 1, 2], [2, 0, 2, 2]]);
  const capture = parseMove(start, [[0, 0, 0, 2], [0, 0, 2, 2]]);
  assert.equal(canSubmit(applyMove(applyMove(start, advance), king)), false);
  const saved = validateAction(start, [capture, king]);
  const exhaustive = [...generateActions(start, { pruneUnsafe: false, cacheMoves: false })];
  const learned = [...generateActions(start)];
  const keys = actions => actions.map(action => positionKey(action.position)).sort();
  assert(exhaustive.length > 0);
  assert.deepEqual(keys(learned), keys(exhaustive));
  assert(learned.some(action => positionKey(action.position) === positionKey(saved)));
  for (const action of learned) validateAction(start, action.moves);
});

test('an active branch may be created before a different board resolves a phantom check', () => {
  const safe = clearBoard(); safe[1][2] = 4;
  const start = position([Array.from({ length: 5 }, () => structuredClone(safe)), [checkedBoard()]]);
  const branch = parseMove(start, [[0, 4, 1, 2], [0, 2, 1, 1]]);
  const middle = applyMove(start, branch);
  assert(raw.boardFuncs.active(middle.board).includes(2));
  assert(inCheck(middle), 'the unfinished older board is still in phantom check');
  assert.equal(canSubmit(middle), false);
  const defense = parseMove(middle, [[1, 0, 0, 0], [1, 0, 0, 1]]);
  const expected = validateAction(start, [branch, defense]);
  const baseline = collect(start, false), pruned = collect(start, true);
  assert.deepEqual(pruned.keys, baseline.keys);
  const matching = pruned.actions.find(action => positionKey(action.position) === positionKey(expected));
  assert(matching, 'branch-then-defense must survive pruning');
  assert.deepEqual(matching.moves, [branch, defense]);
});

function actionIsTactical(start, moves) {
  let current = start, tactical = false;
  for (const move of moves) {
    const target = current.board[move[1][0]][move[1][1]][move[1][2]][move[1][3]];
    const capturesEnemy = target !== 0 && Math.abs(target) % 2 !== current.action % 2;
    const enPassant = move.length === 3;
    const promotesPawn = move[1][4] !== undefined;
    const expected = capturesEnemy || enPassant || promotesPawn;
    assert.equal(isTacticalMove(current, move), expected);
    tactical ||= expected;
    current = applyMove(current, move);
  }
  return tactical;
}

function tacticalEquivalence(start) {
  const exhaustive = [...generateActions(start, { pruneUnsafe: false, cacheMoves: false })];
  const expected = exhaustive.filter(action => actionIsTactical(start, action.moves));
  const tactical = [...generateActions(start, { tacticalOnly: true })];
  const uncached = [...generateActions(start, { tacticalOnly: true, cacheMoves: false })];
  const keys = actions => [...new Set(actions.map(action => positionKey(action.position)))].sort();
  assert.deepEqual(keys(tactical), keys(expected));
  assert.deepEqual(keys(uncached), keys(expected));
  assert(tactical.every(action => actionIsTactical(start, action.moves)));
  return { exhaustive, expected, tactical };
}

test('tactical action filtering preserves quiet components before a capture, including optional future boards', () => {
  const captureBoard = clearBoard(); captureBoard[0][1] = 4; captureBoard[1][2] = 1;
  for (const turn of [0, 2]) {
    const start = position([[clearBoard()], null, Array.from({ length: turn + 1 }, () => structuredClone(captureBoard))]);
    const quiet = parseMove(start, [[0, 0, 0, 0], [0, 0, 0, 1]]);
    const middle = applyMove(start, quiet);
    assert.equal(isTacticalMove(start, quiet), false);
    if (turn === 2) assert(canSubmit(middle), 'capture remains optional after a legal quiet submission');
    const capture = parseMove(middle, [[2, turn, 0, 1], [2, turn, 1, 2]]);
    const expected = validateAction(start, [quiet, capture]);
    const { tactical } = tacticalEquivalence(start);
    const found = tactical.find(action => positionKey(action.position) === positionKey(expected));
    assert(found);
    assert.deepEqual(found.moves, [quiet, capture]);
  }
});

test('tactical action filtering includes a capture that creates a temporal branch', () => {
  const safe = clearBoard(); safe[1][2] = 4;
  const timeline = Array.from({ length: 5 }, () => structuredClone(safe));
  timeline[2][1][1] = 1;
  const start = position([timeline, [clearBoard()]]);
  const branch = parseMove(start, [[0, 4, 1, 2], [0, 2, 1, 1]]);
  assert(isTacticalMove(start, branch));
  const middle = applyMove(start, branch);
  assert(middle.board[2]);
  const quiet = parseMove(middle, [[1, 0, 0, 0], [1, 0, 0, 1]]);
  const expected = validateAction(start, [branch, quiet]);
  const { tactical } = tacticalEquivalence(start);
  assert(tactical.some(action => positionKey(action.position) === positionKey(expected)));
});

test('tactical action filtering treats en passant and quiet promotions as tactical', () => {
  const enPassant = createPosition({ pgn: '1. e4 / a6 2. e5 / d5' });
  const epResults = tacticalEquivalence(enPassant);
  assert(epResults.tactical.some(action => action.moves.some(move => move.length === 3)));
  const board = clearBoard(); board[2][1] = 2;
  const promotion = position([[board]]);
  const promoted = tacticalEquivalence(promotion);
  const promotions = promoted.tactical.flatMap(action => action.moves.filter(move => move[1][4] !== undefined));
  assert.equal(promotions.length, 4);
  assert(promotions.every(move => promotion.board[move[1][0]][move[1][1]][move[1][2]][move[1][3]] === 0));
});

test('quiet castling is excluded and a fully quiet action tree can be skipped', () => {
  const start = createPosition({ pgn: '[Board "Custom"]\n[k7/8/8/8/8/8/8/4K*2R*:0:1:w]' });
  const castle = parseMove(start, 'O-O');
  assert.equal(isTacticalMove(start, castle), false);
  const { exhaustive, tactical } = tacticalEquivalence(start);
  assert(exhaustive.some(action => action.moves.some(move => move.length === 4)));
  assert.deepEqual(tactical, []);
});

test('cached geometry matches fresh moves after an arrival makes two source boards historical', () => {
  const first = clearBoard(); first[0][1] = 4;
  const start = position([[first], [clearBoard()], [clearBoard()]]);
  const rootMoves = pseudoMoves(start);
  const arrival = parseMove(start, [[0, 0, 0, 1], [2, 0, 0, 2]]);
  const middle = applyMove(start, arrival);
  assert.equal(middle.board[0].length, 2);
  assert.equal(middle.board[2].length, 2);
  const cached = rootMoves.filter(move => middle.board[move[0][0]].length - 1 === move[0][1]);
  const fresh = pseudoMoves(middle);
  const keys = moves => moves.map(move => JSON.stringify(move)).sort();
  assert.deepEqual(keys(cached), keys(fresh));
  // This king move now branches into timeline 0, although the identical move
  // coordinates referred to a latest destination at the action's beginning.
  const continuation = parseMove(middle, [[1, 0, 0, 0], [0, 0, 1, 1]]);
  const completed = validateAction(start, [arrival, continuation]);
  assert(completed.board[4]);
  const uncached = [...generateActions(start, { cacheMoves: false })];
  const cachedActions = [...generateActions(start, { cacheMoves: true })];
  const outcomes = actions => actions.map(action => positionKey(action.position)).sort();
  assert.deepEqual(outcomes(cachedActions), outcomes(uncached));
  assert(cachedActions.some(action => positionKey(action.position) === positionKey(completed)));
});

test('source-indexed geometry preserves move order, policy omissions and traversal ticks', () => {
  const first = clearBoard(); first[0][1] = 4;
  const capture = clearBoard(); capture[0][1] = 4; capture[1][2] = 1;
  const positions = [
    createPosition(),
    position([[first], [clearBoard()], [clearBoard()]]),
    position([[first], null, [capture, structuredClone(capture), structuredClone(capture)]]),
  ];
  for (const start of positions) for (const skipOptionalSpatial of [false, true]) for (const tacticalOnly of [false, true]) {
    const run = cacheMoves => {
      let ticks = 0, skipped = 0;
      const actions = [...generateActions(start, {
        cacheMoves, skipOptionalSpatial, tacticalOnly,
        tick: () => ticks++, onSkipOptionalSpatial: () => skipped++,
        orderMoves: (_current, moves) => moves.toReversed(),
      })];
      return { actions, ticks, skipped };
    };
    assert.deepEqual(run(true), run(false), 'indexing may not change order, legality, work budgets or policy evidence');
  }
});

test('a preferred full turn precedes legal prefixes without changing the exhaustive action set', () => {
  const captureBoard = clearBoard(); captureBoard[0][1] = 4; captureBoard[1][2] = 1;
  const start = position([[clearBoard()], null, Array.from({ length: 3 }, () => structuredClone(captureBoard))]);
  const quiet = parseMove(start, [[0, 0, 0, 0], [0, 0, 0, 1]]);
  const middle = applyMove(start, quiet);
  assert(canSubmit(middle), 'the preferred action has a shorter legal prefix');
  const capture = parseMove(middle, [[2, 2, 0, 1], [2, 2, 1, 2]]);
  const preferredAction = [quiet, capture];
  for (const tacticalOnly of [false, true]) {
    const baseline = [...generateActions(start, { tacticalOnly })];
    const ordered = [...generateActions(start, {
      preferredAction, tacticalOnly,
      orderMoves: (_current, moves) => moves.toReversed(),
    })];
    assert.deepEqual(ordered[0].moves, preferredAction);
    const keys = actions => actions.map(action => positionKey(action.position)).sort();
    assert.deepEqual(keys(ordered), keys(baseline));
    assert.equal(new Set(keys(ordered)).size, ordered.length);
    for (const action of ordered) validateAction(start, action.moves);
  }
});

test('stale preferred turns are ignored and quiet preferences cannot enter tactical generation', () => {
  const start = createPosition();
  const baseline = [...generateActions(start)];
  const first = baseline[0].moves;
  const stale = [...first, ...first]; // The source board is already consumed.
  const ordered = [...generateActions(start, { preferredAction: stale })];
  assert.deepEqual(ordered, baseline);
  assert.deepEqual([...generateActions(start, { preferredAction: first, tacticalOnly: true })], []);
});

test('an optional spatial advance can prepare a branch for a later temporal move', () => {
  const first = clearBoard(); first[1][1] = 4;
  const start = position([[first], null, [clearBoard(), clearBoard(), clearBoard()]]);
  assert.deepEqual(raw.boardFuncs.present(start.board, start.action), [0]);
  const advance = parseMove(start, [[2, 2, 0, 0], [2, 2, 0, 1]]);
  const travel = parseMove(start, [[0, 0, 1, 1], [2, 2, 1, 1]]);
  const merged = validateAction(start, [travel]);
  const branched = validateAction(start, [advance, travel]);
  assert.equal(merged.board[4], undefined);
  assert.equal(branched.board[4][3][1][1], 4);
  assert.equal(branched.board[2][3][0][1], 12);
  assert.notEqual(positionKey(merged), positionKey(branched));
  // This is a legality/branch-semantics safeguard, not a claim that this
  // composed position proves a strategic win from the extra branch.
  const expected = positionKey(branched);
  assert([...generateActions(start)].some(action => positionKey(action.position) === expected));
});
