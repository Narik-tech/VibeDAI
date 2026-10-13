import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { analyze } from '../src/search.js';
import {
  applyMove, canSubmit, createPosition, generateActions, inCheck, parseMove,
  positionKey, validateAction,
} from '../src/rules.js';

const root = new URL('../examples/tactics/', import.meta.url);
const rewindPgn = await readFile(new URL('locked-king-rewind.5dpgn', root), 'utf8');
const compoundPgn = await readFile(new URL('locked-king-compound-mate.5dpgn', root), 'utf8');
const lateMatePgn = await readFile(new URL('locked-king-compound-late-mate.5dpgn', root), 'utf8');
const optionalDefensePgn = await readFile(new URL('locked-king-optional-defense.5dpgn', root), 'utf8');
const prefix = '[Board "Custom"]\n[k7/pn6/K7/8/8/8/6PB/8:0:1:w]\n1. Bg1 / Kb8 2. Bd4 / Kc8 3. Bc3';

// Terminal assertions deliberately use complete rules, without unsafe-tree
// pruning, cached moves, or the search policy's optional-board restriction.
function firstReply(position) {
  let work = 0;
  const replies = generateActions(position, {
    pruneUnsafe: false, cacheMoves: false,
    tick() { assert(++work <= 100000, 'terminal verification exceeded its work allowance'); },
  });
  try { return replies.next(); }
  finally { replies.return(); }
}

test('the locked-king line retains all seventeen, twenty-two, and ten Black defenses', () => {
  for (const [suffix, count] of [['', 17], [' / Nc5+ 4. Kxa7', 22], [' / Na5 4. Kxa5', 10]]) {
    const position = createPosition({ pgn: prefix + suffix });
    const actions = [...generateActions(position, { pruneUnsafe: false, cacheMoves: false })];
    assert.equal(actions.length, count);
    for (const action of actions) validateAction(position, action.moves);
  }
});

test('Bc3 preserves the four-turn bishop rewind to c7; Bf6 does not', () => {
  const position = createPosition({ pgn: rewindPgn });
  const control = createPosition({ pgn: rewindPgn.replace('3. Bc3', '3. Bf6') });
  assert(inCheck(position));
  assert(inCheck(control));
  const jump = [[[0, 8, 2, 2], [0, 0, 6, 2]]];
  const after = validateAction(position, jump);
  assert.equal(after.board[2][1][6][2], 4, 'the bishop lands on the new c7 branch');
  assert.deepEqual(after.board[0][0], position.board[0][0], 'the original T1 board remains historical');
  assert.throws(() => parseMove(control, '(0T5)Bf6>>(0T1)c7'));
  assert.equal([...generateActions(position, { pruneUnsafe: false, cacheMoves: false })].length, 4);
  assert.equal([...generateActions(control, { pruneUnsafe: false, cacheMoves: false })].length, 4);

  const final = validateAction(validateAction(after, ['(1T1)Nd8']), ['(1T2)Bxd8']);
  assert(inCheck(final));
  assert.equal(firstReply(final).done, true);
});

test('search finds the c7 rewind and returns a legal mating continuation', () => {
  const position = createPosition({ pgn: rewindPgn });
  const before = positionKey(position);
  const result = analyze(position, {
    unlimitedTime: true, maxDepth: 3, quiescenceDepth: 2, maxNodes: 50000,
  });
  assert.equal(result.mateIn, 3);
  assert.deepEqual(result.bestAction, [[[0, 8, 2, 2], [0, 0, 6, 2]]]);
  let final = position;
  for (const action of result.pv) final = validateAction(final, action);
  assert(inCheck(final));
  assert.equal(firstReply(final).done, true);
  assert.equal(positionKey(position), before);
  assert.equal(result.nodes, result.searchNodes + result.generationNodes);
});

test('the compound bishop mate removes the cross-timeline knight defense', () => {
  const position = createPosition({ pgn: compoundPgn });
  const capture = parseMove(position, '(-1T2)Bg1xa7');
  assert.equal(canSubmit(applyMove(position, capture)), false, 'both present boards require a move');
  const mate = validateAction(position, ['(-1T2)Bg1xa7', '(1T2)Bc7d8']);
  assert(inCheck(mate));
  assert.equal(firstReply(mate).done, true);

  // A king move also completes this component, but leaves the knight's
  // cross-timeline capture available after the other bishop reaches d8.
  const control = validateAction(position, ['(-1T2)Ka6b5', '(1T2)Bc7d8']);
  assert(inCheck(control));
  const defense = validateAction(control, ['(-1T2)Nd7>x(1T2)d8']);
  assert.equal(defense.action, control.action + 1);
  assert.equal(firstReply(control).done, false);
});

test('search proves a complete two-board mating turn within one thousand work nodes', () => {
  const position = createPosition({ pgn: compoundPgn });
  const result = analyze(position, {
    unlimitedTime: true, maxDepth: 2, quiescenceDepth: 0, maxNodes: 1000,
  });
  assert.equal(result.mateIn, 1);
  assert.equal(result.bestAction.length, 2);
  const after = validateAction(position, result.bestAction);
  assert(inCheck(after));
  assert.equal(firstReply(after).done, true);
  assert.equal(result.nodes, result.searchNodes + result.generationNodes);
});

test('the late compound mate requires the bishop component after capturing the temporal knight', () => {
  const position = createPosition({ pgn: lateMatePgn });
  const capture = parseMove(position, '(-1T2)Kxb5');
  assert.equal(canSubmit(applyMove(position, capture)), false);
  const mate = validateAction(position, ['(-1T2)Kxb5', '(1T2)Bd8']);
  assert(inCheck(mate));
  assert.equal(firstReply(mate).done, true);
  const control = validateAction(position, ['(-1T2)Kxb5', '(1T2)Bce5']);
  assert.equal(firstReply(control).done, false);
});

test('an optional Black move preserves the temporal king escape until the bishop pair mates', () => {
  const position = createPosition({ pgn: optionalDefensePgn });
  const control = validateAction(position, ['(-1T2)Kxb5', '(1T2)Bd8']);
  assert(inCheck(control));
  validateAction(control, ['(-1T2)Ka8>>(0T1)b8']);
  assert.equal(firstReply(control).done, false);

  const mate = validateAction(position, ['(1T2)Bhe5', '(-1T2)Bxa7']);
  assert(inCheck(mate));
  assert.equal(firstReply(mate).done, true);
});

test('bounded mate search resolves the optional-defense puzzle before twenty thousand work nodes', () => {
  const position = createPosition({ pgn: optionalDefensePgn });
  const before = positionKey(position);
  const limits = {
    unlimitedTime: true, maxDepth: 2, quiescenceDepth: 2, maxNodes: 20000,
  };
  // Without the shallow full-turn scan, the first checked continuation can
  // consume the budget before the winning bishop component is considered.
  const withoutProbe = analyze(position, { ...limits, mateProbeWork: 0 });
  assert.equal(withoutProbe.completed, false);
  assert.notEqual(withoutProbe.mateIn, 1);
  assert.equal(withoutProbe.stoppedReason, 'nodes');
  const result = analyze(position, limits);
  assert.equal(result.completed, true);
  assert.equal(result.mateIn, 1);
  assert.equal(result.stoppedReason, 'mate');
  const after = validateAction(position, result.bestAction);
  assert(inCheck(after));
  assert.equal(firstReply(after).done, true);
  assert.equal(positionKey(position), before);
  assert(result.nodes <= 20000);
  assert.equal(result.nodes, result.searchNodes + result.generationNodes);
});
