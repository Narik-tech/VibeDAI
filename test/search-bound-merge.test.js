import test from 'node:test';
import assert from 'node:assert/strict';
import { SearchCache } from '../src/search-cache.js';
import { createSearchSession, MATE_SCORE } from '../src/search.js';
import { createPosition, validateAction } from '../src/rules.js';

const action = [[[0, 0, 1, 0], [0, 0, 2, 0]]];
const otherAction = [[[0, 0, 1, 1], [0, 0, 2, 1]]];
const bound = (flag, extra = {}) => ({ depth: 3, quiescenceDepth: 2, score: 42,
  flag, pv: flag === 'lower' ? [action, otherAction] : [otherAction],
  bestAction: flag === 'lower' ? action : otherAction, ...extra });

test('opposite bounds at the same score retain an exact lower-bound PV in either order', () => {
  for (const flags of [['lower', 'upper'], ['upper', 'lower']]) {
    for (const score of [42, MATE_SCORE - 4, -MATE_SCORE + 4]) {
      const cache = new SearchCache(10, 100000);
      const entries = flags.map(flag => Object.freeze(bound(flag, { score })));
      for (const entry of entries) assert(cache.store('position', entry));
      const result = cache.get('position'), lower = entries.find(entry => entry.flag === 'lower');
      assert.equal(result.flag, 'exact');
      assert.equal(result.score, score, 'stored mate scores must stay normalized');
      assert.equal(result.pv, lower.pv);
      assert.equal(result.bestAction, lower.bestAction);
      assert.equal(cache.size, 1);
      assert.equal(entries[0].flag, flags[0], 'merging must not mutate a caller-owned entry');
      assert.equal(entries[1].flag, flags[1]);
    }
  }
});

test('different depths, tactical horizons, scores, or bound directions never become exact', () => {
  for (const difference of [{ depth: 4 }, { quiescenceDepth: 3 }, { quiescenceDepth: undefined },
    { score: 43 }, { flag: 'lower' }]) {
    const cache = new SearchCache(10, 100000);
    cache.store('position', bound('lower'));
    const next = bound('upper', difference);
    assert(cache.store('position', next));
    assert.equal(cache.get('position'), next);
    assert.notEqual(cache.get('position').flag, 'exact');
  }
});

test('tactical bounds merge within each horizon including the negative evasion boundary', () => {
  const cache = new SearchCache(10, 100000);
  for (const horizon of [-1, 0, 1, 2]) {
    const lower = bound('lower', { depth: horizon, quiescenceDepth: undefined });
    cache.store('position', lower, horizon);
    cache.store('position', bound('upper', { depth: horizon, quiescenceDepth: undefined }), horizon);
    assert.equal(cache.get('position', horizon).flag, 'exact');
    assert.equal(cache.get('position', horizon).pv, lower.pv);
  }
  assert.equal(cache.get('position'), undefined);
  assert.equal(cache.size, 4);

  const isolated = new SearchCache(10, 100000);
  isolated.store('position', bound('lower'), 0);
  isolated.store('position', bound('upper'), 1);
  assert.equal(isolated.get('position', 0).flag, 'lower');
  assert.equal(isolated.get('position', 1).flag, 'upper');
});

test('merged entries charge the retained lower-bound PV and obey the byte limit', () => {
  for (const namespace of [undefined, -1, 2]) {
    const measure = new SearchCache(10, 100000);
    const lower = bound('lower');
    measure.store('position', { ...lower, flag: 'exact' }, namespace);
    for (const flags of [['lower', 'upper'], ['upper', 'lower']]) {
      const cache = new SearchCache(10, measure.memoryBytes);
      for (const flag of flags) assert(cache.store('position', bound(flag), namespace));
      assert.equal(cache.get('position', namespace).flag, 'exact');
      assert.equal(cache.memoryBytes, measure.memoryBytes);
    }
    const small = new SearchCache(10, measure.memoryBytes - 1);
    const upper = bound('upper');
    assert(small.store('position', upper, namespace));
    const before = small.memoryBytes;
    assert.equal(small.store('position', lower, namespace), false);
    assert.equal(small.get('position', namespace), upper);
    assert.equal(small.memoryBytes, before);
  }
});

test('a matching fail-high and full-window retry leave a reusable exact root result', () => {
  const position = createPosition();
  const options = { unlimitedTime: true, maxNodes: 100000, quiescenceDepth: 0 };
  const full = { remaining: 3, alpha: -1000000, beta: 1000000 };
  const reference = createSearchSession(position, { ...options, maxTableEntries: 0 }).root(full);
  const session = createSearchSession(position, options);
  const lower = session.root({ ...full, preferred: reference.pv[0],
    alpha: reference.score - 1, beta: reference.score });
  assert.equal(lower.score, reference.score);
  assert.equal(session.root(full).score, reference.score);
  const before = session.statistics();
  const repeated = session.root(full);
  const after = session.statistics();
  assert.equal(repeated.score, reference.score);
  assert.equal(after.generationNodes, before.generationNodes,
    'equal opposite bounds already prove the full-window value');
  assert.equal(after.searchNodes, before.searchNodes + 1);
  assert.equal(after.ttHits, before.ttHits + 1);
  let current = position;
  for (const move of repeated.pv) current = validateAction(current, move);
});
