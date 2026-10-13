import test from 'node:test';
import assert from 'node:assert/strict';
import { analyze } from '../src/transformer-search.js';
import { applyMove, canSubmit, createPosition, inCheck, validateAction } from '../src/rules.js';

const options = {
  maxDepth: 0, maxNodes: 200_000, timeMs: 10_000,
  evaluateBatch: async positions => positions.map(() => 0),
};

test('dynamic depth starts at one and preserves zero as the requested limit with no budget', async () => {
  for (const budget of [{ maxNodes: 0 }, { timeMs: 0 }]) {
    const result = await analyze(createPosition(), { ...options, ...budget });
    assert.equal(result.depthMode, 'dynamic');
    assert.equal(result.currentMaxDepth, 1);
    assert.equal(result.dynamicDepthThreshold, 3);
    assert.equal(result.limits.maxDepth, 0);
    assert.equal(result.depth, 0);
    assert.equal(result.evaluations, 0);
  }
});

test('dynamic depth compares three root contenders before opening depth two', async () => {
  const reports = [];
  let stop = false;
  const result = await analyze(createPosition(), { ...options,
    onProgress: info => { reports.push(info); stop ||= info.currentMaxDepth === 2; },
    shouldStop: () => stop,
  });
  assert.equal(reports[0].currentMaxDepth, 1);
  assert.equal(result.currentMaxDepth, 2);
  assert.equal(result.rootActionsSearched, 3);
  assert.equal(result.depth, 1);
  assert.equal(result.selectiveDepth, 1);
  assert.equal(result.stoppedReason, 'cancelled');
  assert.equal(result.depthStats.length, 1);
  assert.equal(result.depthStats[0].trueEvaluations, 3);
  assert.ok(result.depthStats[0].candidates > 0, 'pending breadth does not block useful depth');
});

test('opening depth three requires opponent reply coverage under each root contender', async () => {
  const transitions = [];
  let ceiling = 1, stop = false;
  const position = createPosition();
  const result = await analyze(position, { ...options,
    onProgress: info => {
      if (info.currentMaxDepth !== ceiling) {
        assert.equal(info.currentMaxDepth, ceiling + 1);
        assert.ok(info.depthStats.some(level => level.depth === ceiling));
        const contenders = info.rankings[0].entries.slice(0, 3);
        assert.ok(contenders.every(entry => entry.evaluationType === 'true'));
        if (ceiling === 2) {
          assert.ok(contenders.every(entry => entry.searchedReplies >= 2));
          assert.ok(contenders.every(entry => entry.line.length >= 2));
          assert.ok(info.depthStats.some(level => level.candidates > 0));
        }
        ceiling = info.currentMaxDepth;
        transitions.push(ceiling);
      }
      stop ||= ceiling === 3;
    },
    shouldStop: () => stop,
  });
  assert.deepEqual(transitions, [2, 3]);
  assert.equal(result.limits.maxDepth, 0);
  assert.equal(result.currentMaxDepth, 3);
  assert.equal(result.depth, 2);
  assert.equal(result.stoppedReason, 'cancelled');
  let current = position;
  for (const action of result.pv) current = validateAction(current, action);
});

test('small candidate limits advance one ceiling at a time with contender progress', async () => {
  const transitions = [];
  let ceiling = 1, stop = false;
  const result = await analyze(createPosition(), { ...options, candidateLimit: 2, innerCandidateLimit: 2,
    onProgress: info => {
      if (info.currentMaxDepth !== ceiling) {
        assert.equal(info.currentMaxDepth, ceiling + 1);
        assert.ok(info.depth >= ceiling);
        assert.ok(info.rankings[0].entries.every(entry => entry.line.length >= ceiling));
        ceiling = info.currentMaxDepth;
        transitions.push(ceiling);
      }
      stop ||= ceiling === 5;
    },
    shouldStop: () => stop,
  });
  assert.deepEqual(transitions, [2, 3, 4, 5]);
  assert.equal(result.stoppedReason, 'cancelled');
});

test('fixed depth one widens through all opening candidates without raising the ceiling', async () => {
  const result = await analyze(createPosition(), { ...options, maxDepth: 1 });
  assert.equal(result.depthMode, 'fixed');
  assert.equal(result.dynamicDepthThreshold, null);
  assert.equal(result.currentMaxDepth, 1);
  assert.equal(result.depth, 1);
  assert.equal(result.rootActionsSearched, 20);
  assert.ok(result.wideningSteps >= 2);
  assert.equal(result.depthStats[0].candidates, 0);
  assert.equal(result.stoppedReason, 'depth');
});

test('a checking continuation regains normal reply coverage when dynamic depth overtakes its extension', async () => {
  // Rd1+ Kc4 Rc1+ is first reached while the ordinary ceiling is three.
  // Its evasion search initially admits one reply; raising the ceiling must
  // let the same position compare further Black king moves.
  const position = createPosition({ pgn: '[Board "Custom"]\n[Size "4x4"]\n[3k/4/4/KR2:0:1:w]' });
  const extended = new Map();
  let reopened = null, stop = false;
  const result = await analyze(position, { ...options, unlimitedTime: true, maxNodes: 50000,
    candidateLimit: 3, innerCandidateLimit: 4, initialCandidates: 4,
    componentBatchSize: 4, extensionCandidateLimit: 1,
    scoreMoves: async (current, moves) => moves.map(move => {
      const next = applyMove(current, move);
      return canSubmit(next) && inCheck({ ...next, action: next.action + 1 }) ? 100 : 0;
    }),
    onProgress: info => {
      for (const row of info.rankings) for (const entry of row.entries) {
        if (entry.forcing && row.depth >= info.currentMaxDepth && entry.generatedReplies === 1) {
          extended.set(entry.id, row.depth);
        }
        if (extended.has(entry.id) && info.currentMaxDepth > extended.get(entry.id)
          && entry.generatedReplies > 1 && entry.searchedReplies > 1) reopened = entry;
      }
      stop ||= Boolean(reopened) || info.currentMaxDepth >= 5;
    },
    shouldStop: () => stop,
  });
  assert.ok(extended.size > 0, 'a checking branch received the one-reply tactical extension');
  assert.ok(reopened, 'that same branch must admit more replies once inside the ordinary horizon');
  assert.ok(reopened.generatedReplies <= 4);
  assert.ok(reopened.searchedReplies > 1, 'additional replies are evaluated, not merely listed');
  assert.equal(result.stoppedReason, 'cancelled');
  let current = position;
  for (const action of result.pv) current = validateAction(current, action);
});
