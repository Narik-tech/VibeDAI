import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { analyze, MATE_SCORE } from '../src/search.js';
import { analyze as parallelAnalyze } from '../src/parallel-search.js';
import { createPosition, generateActions, inCheck, positionKey, validateAction } from '../src/rules.js';

const pgn = await readFile(new URL('../examples/tactics/locked-king-optional-defense.5dpgn', import.meta.url), 'utf8');
const limits = { unlimitedTime: true, maxDepth: 3, maxNodes: 20000, quiescenceDepth: 2 };

function assertProof(position, result, sign) {
  assert.equal(result.completed, true);
  assert.equal(result.searchPolicy, 'full-rules-mate');
  assert.equal(result.mateProbe.status, 'mate');
  assert.equal(result.score, sign * (MATE_SCORE - 1));
  assert.equal(result.mateIn, sign);
  assert.equal(result.depth, 1);
  assert.equal(result.searchingDepth, 1);
  assert.equal(result.rootActionsSearched, result.mateProbe.candidates);
  assert.equal(result.effectiveQuiescenceDepth, 0);
  assert.equal(result.stoppedReason, 'mate');
  assert.equal(result.pv.length, 1);
  assert.equal(result.nodes, result.searchNodes + result.generationNodes);
  assert(result.mateProbe.work <= result.limits.mateProbeWork);
  assert(result.nodes <= result.limits.maxNodes);
  const next = validateAction(position, result.bestAction);
  assert(inCheck(next));
  const replies = generateActions(next, { firstOnly: true, pruneUnsafe: false, cacheMoves: false });
  try { assert.equal(replies.next().done, true); }
  finally { replies.return(); }
}

test('root proofs retain White and Black score signs and parallel shared work accounting', async () => {
  const white = createPosition({ pgn });
  const black = createPosition({ pgn: '1. f3 / e6 2. Nc3 / Qe7 3. a3' });
  black.board = [black.board[0], null, structuredClone(black.board[0])];
  for (const [position, sign] of [[white, 1], [black, -1]]) {
    const original = positionKey(position);
    const serial = analyze(position, limits);
    assertProof(position, serial, sign);
    for (const threads of [1, 4]) {
      const progress = [];
      const result = await parallelAnalyze(position, { ...limits, threads, onProgress: value => progress.push(value) });
      assertProof(position, result, sign);
      assert.equal(result.threadsUsed, 1, 'an immediate proof should avoid worker startup');
      assert.equal(result.nodes, serial.nodes);
      assert.deepEqual(result.bestAction, serial.bestAction);
      assertProof(position, progress.at(-1), sign);
    }
    assert.equal(positionKey(position), original);
  }
});

test('cancellation inside the root probe preserves a legal fallback without a mate claim', () => {
  const position = createPosition({ pgn });
  for (const budget of [25, 50]) {
    let calls = 0;
    const result = analyze(position, { ...limits, shouldStop: () => ++calls > budget });
    assert.equal(result.nodes, budget);
    assert.equal(result.stoppedReason, 'cancelled');
    assert.equal(result.completed, false);
    assert.equal(result.score, null);
    assert.equal(result.mateIn, null);
    assert.equal(result.searchPolicy, 'present-spatial');
    validateAction(position, result.bestAction);
  }
});

test('a local probe limit remains unknown and lets normal search continue deterministically', () => {
  const position = createPosition({ pgn });
  const options = { ...limits, mateProbeWork: 1 };
  const first = analyze(position, options), second = analyze(position, options);
  assert.equal(first.mateProbe.status, 'unknown');
  assert.equal(first.mateProbe.reason, 'work');
  assert.equal(first.mateProbe.work, 1);
  assert.equal(first.searchPolicy, 'present-spatial');
  assert(first.searchNodes > 0);
  assert.equal(first.nodes, options.maxNodes);
  assert.equal(first.score, second.score);
  assert.deepEqual(first.bestAction, second.bestAction);
  assert.deepEqual(first.mateProbe, second.mateProbe);
  validateAction(position, first.bestAction);
});
