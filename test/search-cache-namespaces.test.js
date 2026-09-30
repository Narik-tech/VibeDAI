import test from 'node:test';
import assert from 'node:assert/strict';
import { SearchCache } from '../src/search-cache.js';

const entry = (depth = 1, flag = 'exact', pv = []) => ({ depth, flag, score: 42, pv });
const action = [[[0, 0, 1, 0], [0, 0, 2, 0]]];

test('default and quiescence namespaces isolate identical position keys', () => {
  const cache = new SearchCache(10, 100000);
  const normal = entry(), quiet = entry(0), tactical = entry(1);
  assert(cache.store('position', normal));
  assert(cache.store('position', quiet, 0));
  assert(cache.store('position', tactical, 1));

  assert.equal(cache.size, 3);
  assert.equal(cache.get('position'), normal);
  assert.equal(cache.get('position', 0), quiet);
  assert.equal(cache.get('position', 1), tactical);
  assert.equal(cache.get('position', 2), undefined);
  assert.equal(cache.get('absent', 0), undefined);

  const replacement = entry(2);
  assert(cache.store('position', replacement, 0));
  assert.equal(cache.size, 3);
  assert.equal(cache.get('position', 0), replacement);
  assert.equal(cache.get('position'), normal);
  assert.equal(cache.get('position', 1), tactical);
});

test('all namespaces share FIFO limits and replacements retain their age', () => {
  const cache = new SearchCache(3, 100000);
  cache.store('same', entry());
  cache.store('same', entry(), 0);
  cache.store('same', entry(), 1);
  const replacement = entry(2);
  cache.store('same', replacement, 0);
  cache.store('new', entry(), 2);
  assert.equal(cache.get('same'), undefined);
  assert.equal(cache.get('same', 0), replacement);
  assert(cache.get('same', 1));
  assert(cache.get('new', 2));

  cache.store('last', entry());
  assert.equal(cache.get('same', 0), undefined,
    'eviction must remove the namespace index as well as the shared record');
  assert(cache.get('same', 1));
  assert(cache.get('new', 2));
  assert(cache.get('last'));
  assert.equal(cache.size, 3);
});

test('namespace storage is charged and growing replacements evict across namespaces', () => {
  const plain = new SearchCache(10, 100000);
  plain.store('a', entry());
  const namespaced = new SearchCache(10, 100000);
  namespaced.store('a', entry(), 0);
  assert(namespaced.memoryBytes > plain.memoryBytes,
    'the additional namespace index must be charged to the memory budget');

  const larger = entry(2, 'exact', [action, action]);
  const measure = new SearchCache(10, 100000);
  measure.store('a', larger, 0);
  const cache = new SearchCache(10, measure.memoryBytes);
  cache.store('a', entry(), 0);
  cache.store('b', entry());
  assert.equal(cache.size, 2);
  assert(cache.store('a', larger, 0));
  assert.equal(cache.get('b'), undefined);
  assert.equal(cache.get('a', 0), larger);
  assert.equal(cache.size, 1);
  assert.equal(cache.memoryBytes, measure.memoryBytes);
});

test('rejected namespace replacements leave both lookup and accounting intact', () => {
  const cache = new SearchCache(10, 1000);
  const original = entry(5, 'lower');
  cache.store('same', original, 0);
  cache.store('same', entry(), 1);
  const bytes = cache.memoryBytes;

  assert.equal(cache.store('same', entry(1, 'upper'), 0), false);
  assert.equal(cache.store('same', entry(6, 'exact', Array(1000).fill(action)), 0), false);
  assert.equal(cache.get('same', 0), original);
  assert(cache.get('same', 1));
  assert.equal(cache.size, 2);
  assert.equal(cache.memoryBytes, bytes);
});

test('bounded namespace evictions accept 64 entries and reject 65 atomically', () => {
  const measure = new SearchCache(1000, 1000000);
  measure.store('000', entry(), 0);
  const entryBytes = measure.memoryBytes;
  for (const needed of [64, 65]) {
    const cache = new SearchCache(1000, 70 * entryBytes);
    for (let index = 0; index < 70; index++) {
      cache.store(String(index).padStart(3, '0'), entry(), index % 2);
    }
    const key = 'x'.repeat(3 + (needed - 1) * entryBytes / 2);
    const replacement = entry();
    assert.equal(cache.store(key, replacement, 0), needed === 64);
    assert.equal(cache.memoryBytes, 70 * entryBytes);
    if (needed === 64) {
      assert.equal(cache.size, 7);
      for (let index = 0; index < 64; index++) {
        assert.equal(cache.get(String(index).padStart(3, '0'), index % 2), undefined);
      }
      assert(cache.get('064', 0));
      assert(cache.get('065', 1));
      assert.equal(cache.get(key, 0), replacement);
    } else {
      assert.equal(cache.size, 70);
      for (let index = 0; index < 70; index++) {
        assert(cache.get(String(index).padStart(3, '0'), index % 2));
      }
      assert.equal(cache.get(key, 0), undefined);
    }
  }
});
