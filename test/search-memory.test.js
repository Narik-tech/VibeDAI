import test from 'node:test';
import assert from 'node:assert/strict';
import { SearchCache } from '../src/search-cache.js';
import { analyze } from '../src/search.js';
import { createPosition, validateAction } from '../src/rules.js';

const entry = (depth = 1, flag = 'exact', pv = []) => ({ depth, flag, score: 42, pv, bestAction: pv[0] });
const action = [[[0, 0, 1, 0], [0, 0, 2, 0]]];

test('cache accounts for full UTF-16 history and retained principal variation', () => {
  const cache = new SearchCache(10, 100000);
  const small = entry();
  cache.store('a', small);
  const firstBytes = cache.memoryBytes;
  cache.store('abcd', entry());
  assert.equal(cache.memoryBytes, firstBytes * 2 + 6);
  cache.store('moves', entry(2, 'exact', [action, action]));
  assert(cache.memoryBytes > firstBytes * 3 + 6);
  assert.equal(cache.get('a'), small);
  assert.equal(cache.size, 3);
});

test('replacement accounting preserves deep bounds and oversized existing entries', () => {
  const cache = new SearchCache(10, 1000);
  const original = entry(5, 'lower');
  assert(cache.store('a', original));
  const initialBytes = cache.memoryBytes;
  assert.equal(cache.store('a', entry(1, 'upper')), false);
  assert.equal(cache.memoryBytes, initialBytes);
  assert.equal(cache.get('a'), original);
  assert.equal(cache.store('a', entry(6, 'exact', Array(1000).fill(action))), false);
  assert.equal(cache.get('a'), original);
  assert.equal(cache.memoryBytes, initialBytes);
  assert(cache.store('a', entry(1, 'exact', [action])));
  assert(cache.memoryBytes > initialBytes);
  assert(cache.store('a', entry(2, 'exact')));
  assert.equal(cache.memoryBytes, initialBytes);
  assert.equal(cache.size, 1);
});

test('memory and entry limits evict FIFO while replacements keep their age', () => {
  const measure = new SearchCache(10, 10000);
  measure.store('a', entry());
  for (const cache of [new SearchCache(2, 10000), new SearchCache(10, measure.memoryBytes * 2)]) {
    cache.store('a', entry());
    cache.store('b', entry());
    cache.store('a', entry(2));
    cache.store('c', entry());
    assert.equal(cache.get('a'), undefined);
    assert(cache.get('b'));
    assert(cache.get('c'));
    assert.equal(cache.size, 2);
    assert(cache.memoryBytes <= cache.maxBytes);
  }
});

test('a growing replacement evicts other entries and counts its old bytes once', () => {
  const measure = new SearchCache(10, 10000);
  measure.store('a', entry(2, 'exact', [action, action]));
  const cache = new SearchCache(10, measure.memoryBytes);
  cache.store('a', entry());
  cache.store('b', entry());
  assert.equal(cache.size, 2);
  assert(cache.store('a', entry(2, 'exact', [action, action])));
  assert.equal(cache.get('b'), undefined);
  assert.equal(cache.size, 1);
  assert.equal(cache.memoryBytes, measure.memoryBytes);
});

test('oversized histories and costly bulk evictions leave the cache intact', () => {
  const cache = new SearchCache(1000, 40000);
  for (let i = 0; i < 100; i++) cache.store(String(i), entry());
  const priorBytes = cache.memoryBytes;
  assert.equal(cache.store('x'.repeat(30000), entry()), false);
  assert.equal(cache.store('x'.repeat(19000), entry()), false);
  assert.equal(cache.size, 100);
  assert.equal(cache.memoryBytes, priorBytes);
});

test('bounded eviction accepts the last allowed entry and rejects a larger batch atomically', () => {
  const measure = new SearchCache(1000, 1000000);
  measure.store('000', entry());
  const entryBytes = measure.memoryBytes;
  for (const needed of [64, 65]) {
    const cache = new SearchCache(1000, 70 * entryBytes);
    for (let i = 0; i < 70; i++) cache.store(String(i).padStart(3, '0'), entry());
    // Vary only the retained UTF-16 history, so the new entry needs precisely
    // the requested number of equally sized entries to be evicted.
    const key = 'x'.repeat(3 + (needed - 1) * entryBytes / 2);
    const value = entry();
    assert.equal(cache.store(key, value), needed === 64);
    assert.equal(cache.memoryBytes, 70 * entryBytes);
    if (needed === 64) {
      assert.equal(cache.size, 7);
      assert.equal(cache.get('063'), undefined);
      assert(cache.get('064'));
      assert.equal(cache.get(key), value);
    } else {
      assert.equal(cache.size, 70);
      for (let i = 0; i < 70; i++) assert(cache.get(String(i).padStart(3, '0')));
      assert.equal(cache.get(key), undefined);
    }
  }
});

test('zero byte or entry budgets disable retained cache data', () => {
  for (const cache of [new SearchCache(0, 1000), new SearchCache(1000, 0)]) {
    assert.equal(cache.store('a', entry()), false);
    assert.equal(cache.size, 0);
    assert.equal(cache.memoryBytes, 0);
  }
});

test('search memory budgets preserve exact results and cap every progress snapshot', () => {
  const position = createPosition({ pgn: '[Board "Custom"]\n[k7/pn6/K7/8/8/8/6PB/8:0:1:w]' });
  const options = { timeMs: 10000, maxNodes: 200000, maxDepth: 2, quiescenceDepth: 2 };
  const reference = analyze(position, { ...options, cacheMemoryMb: 0 });
  assert.equal(reference.depth, 2);
  assert.equal(reference.tableEntries, 0);
  assert.equal(reference.cacheMemoryBytes, 0);
  assert.equal(reference.ttHits, 0);
  for (const cacheMemoryMb of [0.001, 0.01, 128]) {
    const snapshots = [];
    const result = analyze(position, { ...options, cacheMemoryMb, onProgress: state => snapshots.push(state) });
    assert.equal(result.depth, 2);
    assert.equal(result.score, reference.score);
    assert.deepEqual(result.bestAction, reference.bestAction);
    let current = position;
    for (const action of result.pv) current = validateAction(current, action);
    for (const snapshot of [...snapshots, result]) {
      assert.equal(snapshot.limits.cacheMemoryMb, cacheMemoryMb);
      assert.equal(snapshot.limits.maxTableEntries, 100000);
      assert(snapshot.cacheMemoryBytes <= Math.floor(cacheMemoryMb * 1024 * 1024));
      assert(snapshot.cacheMemoryBytes >= 0);
    }
    if (cacheMemoryMb === 128) assert(result.cacheMemoryBytes > 0);
  }
});

test('memory options normalize at the API boundary without allocating the budget', () => {
  const position = createPosition();
  for (const [cacheMemoryMb, expected] of [[undefined, 128], [-1, 0], [Infinity, 128], [9000, 4096]]) {
    const result = analyze(position, { cacheMemoryMb, maxNodes: 0 });
    assert.equal(result.limits.cacheMemoryMb, expected);
    assert.equal(result.cacheMemoryBytes, 0);
    assert.equal(result.tableEntries, 0);
  }
});
