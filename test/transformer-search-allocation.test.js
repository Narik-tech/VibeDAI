import test from 'node:test';
import assert from 'node:assert/strict';
import { canDeepen, chooseWork, rankDepths } from '../src/transformer-frontier.js';

function tree(rootSign = 1, rootOptions = {}) {
  let index = 0;
  const root = { depth: 0, parent: null, children: [], best: null, visits: 1,
    canWiden: false, generationDone: true, limit: 64, ...rootOptions };
  const levels = [[]];
  function add(parent, score, extra = {}) {
    const value = score * rootSign;
    const node = { depth: parent.depth + 1, parent, index: index++,
      candidateScore: value, trueScore: value, value, children: null, best: null,
      terminal: null, mateProven: false, visits: 0, canWiden: false,
      generationDone: false, ...extra };
    (parent.children ??= []).push(node);
    (levels[node.depth] ??= []).push(node);
    if (node.trueScore !== null) parent.best ??= node;
    return node;
  }
  const rankings = () => rankDepths(levels, rootSign);
  return { root, add,
    ready: maxDepth => canDeepen(rankings(), maxDepth, { root, rootSign }),
    choose: maxDepth => chooseWork(rankings(), { root, rootSign, maxDepth }),
  };
}

test('an unfinished small root batch widens before opening depth or expanding a contender', () => {
  for (const rootSign of [1, -1]) for (const admitted of [1, 2]) {
    const graph = tree(rootSign, { canWiden: true, generationDone: false, limit: 8 });
    for (let i = 0; i < admitted; i++) graph.add(graph.root, 100 - i);
    assert.equal(graph.ready(1), false,
      'a pending generator must supply the initial contender coverage');
    assert.deepEqual(graph.choose(2), { kind: 'widen', node: graph.root },
      'fixed-depth search needs the same root coverage before reply search');
  }
});

test('root coverage respects its cap and accepts an exhausted short generator', () => {
  for (const limit of [1, 2, 3, 8]) {
    const admitted = Math.min(limit, 3);
    const graph = tree(1, { limit, canWiden: admitted < limit, generationDone: admitted === limit });
    for (let i = 0; i < admitted; i++) graph.add(graph.root, 100 - i);
    assert.equal(graph.ready(1), true, `root cap ${limit}`);
  }
  const exhausted = tree(1, { limit: 64 });
  exhausted.add(exhausted.root, 100);
  assert.equal(exhausted.ready(1), true, 'one legal candidate can exhaust a much larger cap');
});

test('an exhausted selective dead end does not block a live contender from deepening', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign);
    const closed = graph.add(graph.root, 100, {
      children: [], generationDone: true, exhaustive: false,
    });
    const live = graph.add(graph.root, 90);
    graph.add(live, 90);
    graph.add(live, 100);
    assert.equal(graph.ready(2), true,
      'a closed selective branch has no available work at the current ceiling');
    assert.equal(closed.terminal, null, 'frontier exhaustion is not a terminal proof');
    assert.equal(closed.mateProven, false);
  }
});

test('a closed principal tip still waits for available sibling work below the ceiling', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign);
    const contender = graph.add(graph.root, 100);
    graph.add(contender, 100, { children: [], generationDone: true, exhaustive: false });
    const openReply = graph.add(contender, 110);
    const live = graph.add(graph.root, 90);
    const reply = graph.add(live, 90);
    graph.add(live, 100);
    graph.add(reply, 90);
    assert.equal(graph.ready(3), false,
      'an exhausted best line must not hide expandable alternatives in its subtree');
    assert.deepEqual(graph.choose(3), { kind: 'expand', node: openReply });
  }
});

test('exploration follows branch visits when root and branch work counts differ', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign, { visits: 1 });
    const contender = graph.add(graph.root, 100, { visits: 8 });
    graph.add(contender, 100, { visits: 20 });
    const overlooked = graph.add(contender, 1100, { visits: 0 });
    assert.deepEqual(graph.choose(4), { kind: 'expand', node: overlooked },
      'a branch earns exploration after its own work, independent of root modulo');
  }
});

test('selection stops before traversing lower branches after finding preferred work', () => {
  const graph = tree();
  const roots = [100, 90, 80, -1000].map(score => graph.add(graph.root, score));
  const replies = roots.map((parent, index) => [
    graph.add(parent, [100, 90, 80, -1000][index]),
    graph.add(parent, [110, 100, 90, -990][index]),
  ]);
  let lowerChildrenReads = 0;
  const lowerChildren = roots[3].children;
  Object.defineProperty(roots[3], 'children', {
    get() { lowerChildrenReads++; return lowerChildren; },
  });
  assert.deepEqual(graph.choose(4), { kind: 'expand', node: replies[0][0] });
  assert.equal(lowerChildrenReads, 0,
    'the lower subtree should remain untouched once a preferred frontier is available');
});

test('dynamic depth follows competitive roots after every contender gets initial reply coverage', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign);
    const roots = [100, 90, -1000].map(score => graph.add(graph.root, score));
    const replies = roots.map((parent, index) => [
      graph.add(parent, [100, 90, -1000][index]),
      graph.add(parent, [110, 100, -990][index]),
    ]);
    graph.add(replies[0][0], 100);
    graph.add(replies[0][0], 90);
    graph.add(replies[1][0], 90);
    graph.add(replies[1][0], 80);
    assert.equal(graph.ready(3), true,
      'a clearly losing third root need not match the competitive lines in depth');
    replies[2][1].trueScore = null;
    assert.equal(graph.ready(3), false,
      'initial defensive coverage still applies to the losing third contender');
    replies[2][1].trueScore = replies[2][1].value;
    roots[2].trueScore = roots[2].value = 0;
    replies[2][0].trueScore = replies[2][0].value = 0;
    replies[2][1].trueScore = replies[2][1].value = 10 * rootSign;
    assert.equal(graph.ready(3), false,
      'a newly competitive shallow root must catch up before the ceiling increases');
  }
});

test('a live alternative at the ceiling can deepen after its preferred line closes', () => {
  for (const rootSign of [1, -1]) for (const closure of ['selective', 'stalemate']) {
    const graph = tree(rootSign, { limit: 1 });
    const contender = graph.add(graph.root, 0);
    graph.add(contender, 0, { children: [], generationDone: true, exhaustive: false,
      terminal: closure === 'stalemate' ? { score: 0, mateProven: true } : null,
    });
    const alternative = graph.add(contender, 10);
    graph.add(alternative, 10);
    graph.add(alternative, 0);
    assert.equal(graph.choose(3), null, 'all work under the current ceiling is complete');
    assert.equal(graph.ready(3), true,
      `the short ${closure} PV must not hide a live alternative at the ceiling`);
  }
});

test('known selective closures at the ceiling do not open an unnecessary depth', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign, { limit: 1 });
    const contender = graph.add(graph.root, 0);
    const closed = [0, 10].map(score => graph.add(contender, score, {
      children: [], generationDone: true, exhaustive: false, terminal: false,
    }));
    assert.equal(graph.choose(2), null);
    assert.equal(graph.ready(2), false, 'there is no live continuation beyond this ceiling');
    assert.ok(closed.every(node => node.terminal === false && node.mateProven === false),
      'selective closure must not become a terminal certificate');
  }
});

test('progressive widening scales work to the configured initial width', () => {
  const graph = tree(1, {
    initialWidth: 32, canWiden: true, generationDone: false, limit: 128,
  });
  const roots = Array.from({ length: 32 }, (_, index) => graph.add(graph.root, 100 - index));
  for (const [index, contender] of roots.slice(0, 3).entries()) {
    graph.add(contender, 100 - index);
    graph.add(contender, 110 - index);
  }
  for (const visits of [1, 3, 24, 95]) {
    graph.root.visits = visits;
    assert.equal(graph.choose(4).kind, 'expand',
      `a 32-wide root should spend work below its candidates at visit ${visits}`);
  }
  graph.root.visits = 97;
  assert.deepEqual(graph.choose(4), { kind: 'widen', node: graph.root },
    'after enough local work, the configured width can double');
});
