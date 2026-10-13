import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseWork, rankDepths } from '../src/transformer-frontier.js';

function tree(rootSign = 1) {
  let nextIndex = 0;
  const root = { depth: 0, children: [], best: null, visits: 2 };
  const levels = [[]];
  function add(parent, score, isTrue = true, extra = {}) {
    const value = score * rootSign;
    const node = {
      parent, depth: parent.depth + 1, index: nextIndex++,
      candidateScore: value, trueScore: isTrue ? value : null, value: isTrue ? value : null,
      children: null, best: null, terminal: null, mateProven: false, visits: 0, ...extra,
    };
    (parent.children ??= []).push(node);
    (levels[node.depth] ??= []).push(node);
    if (isTrue) parent.best ??= node;
    return node;
  }
  function chain(parent, length, score) {
    const nodes = [];
    for (let i = 0; i < length; i++) {
      parent = add(parent, score);
      nodes.push(parent);
    }
    return nodes;
  }
  return { root, levels, add, chain,
    rankings: () => rankDepths(levels, rootSign),
    choose: (maxDepth = 8) => chooseWork(rankDepths(levels, rootSign), { maxDepth, root, rootSign }),
  };
}

test('competitive short sidelines get catch-up work without changing displayed ranks', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign);
    const leader = graph.chain(graph.root, 5, 100);
    const sideline = graph.chain(graph.root, 2, 90);
    const order = graph.rankings().map(level => level.ranked.slice());
    assert.deepEqual(graph.choose(), { kind: 'expand', node: sideline.at(-1) });
    assert.deepEqual(graph.rankings().map(level => level.ranked), order);
    assert.equal(graph.rankings()[0].ranked[0], leader[0]);
  }
});

test('catch-up uses a bounded share of work instead of dominating every operation', () => {
  const graph = tree();
  const leader = graph.chain(graph.root, 5, 100);
  const sideline = graph.chain(graph.root, 2, 90);
  for (const workCount of [1, 3, 4, 5, 7]) {
    graph.root.visits = workCount;
    assert.deepEqual(graph.choose(), { kind: 'expand', node: leader.at(-1) });
  }
  for (const workCount of [2, 6, 10]) {
    graph.root.visits = workCount;
    assert.deepEqual(graph.choose(), { kind: 'expand', node: sideline.at(-1) });
  }
});

test('a losing-looking short sideline has no automatic catch-up claim', () => {
  const graph = tree();
  const leader = graph.chain(graph.root, 5, 100);
  graph.chain(graph.root, 2, -1000);
  assert.deepEqual(graph.choose(), { kind: 'expand', node: leader.at(-1) });
});

test('forcing and unstable sidelines retain catch-up eligibility despite their score gap', () => {
  for (const reason of ['forcing', 'unstable']) {
    const graph = tree();
    graph.chain(graph.root, 5, 100);
    const sideline = graph.chain(graph.root, 2, -1000);
    if (reason === 'forcing') sideline.at(-1).forcing = true;
    else sideline[0].trueScore = 1000;
    assert.deepEqual(graph.choose(), { kind: 'expand', node: sideline.at(-1) }, reason);
  }
});

test('resolved checks and old evaluation corrections lose their extra catch-up priority', () => {
  for (const reason of ['resolved-check', 'old-correction']) {
    const graph = tree();
    const leader = graph.chain(graph.root, 5, 100);
    const sideline = graph.chain(graph.root, 2, -1000);
    if (reason === 'resolved-check') sideline[0].forcing = true;
    else {
      sideline[0].trueScore = 1000;
      sideline[0].visits = 400;
    }
    assert.deepEqual(graph.choose(), { kind: 'expand', node: leader.at(-1) }, reason);
  }
});

test('exploration gives weak sidelines a separate opportunity', () => {
  const graph = tree();
  const leader = graph.chain(graph.root, 5, 100);
  const sideline = graph.chain(graph.root, 2, -1000);
  leader[0].visits = 20;
  graph.root.visits = 8;
  assert.deepEqual(graph.choose(), { kind: 'expand', node: sideline.at(-1) });
});

test('zero- and one-turn differences do not claim catch-up work', () => {
  for (const leaderLength of [3, 4]) {
    const graph = tree();
    const leader = graph.chain(graph.root, leaderLength, 100);
    graph.chain(graph.root, 3, 90);
    assert.deepEqual(graph.choose(), { kind: 'expand', node: leader.at(-1) });
  }
});

test('opponent reply sidelines compare scores from the opponent perspective', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign);
    const parent = graph.add(graph.root, 100);
    parent.visits = 2;
    graph.chain(parent, 4, 100);
    const sideline = graph.add(parent, 110);
    assert.deepEqual(graph.choose(), { kind: 'expand', node: sideline });
    sideline.trueScore = sideline.value = 1000 * rootSign;
    const work = graph.choose();
    assert.notEqual(work.node, sideline, 'a poor opponent reply receives no score-based catch-up');
  }
});

test('a shorter higher-ranked line suppresses catch-up throughout demoted branches', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign);
    const leader = graph.chain(graph.root, 3, 100);
    const lower = graph.chain(graph.root, 7, 90);
    graph.chain(graph.root, 4, 80);
    graph.add(lower[3], 80);
    assert.equal(graph.rankings()[3].ranked[0], lower[3],
      'a lower branch can become first-ranked at a depth where the root leader has no node');
    assert.deepEqual(graph.choose(), { kind: 'expand', node: leader.at(-1) });
  }
});

test('newly promoted short leaders revoke a demoted branch catch-up claim', () => {
  const graph = tree();
  graph.chain(graph.root, 7, 100);
  const sideline = graph.chain(graph.root, 4, 90);
  const short = graph.chain(graph.root, 3, 80);
  assert.deepEqual(graph.choose(), { kind: 'expand', node: sideline.at(-1) });
  short[0].value = 110;
  assert.deepEqual(graph.choose(), { kind: 'expand', node: short.at(-1) });
});

test('resolved short lines and closed sidelines do not block available work', () => {
  for (const closed of ['terminal', 'proven', 'expanded']) {
    const graph = tree();
    const leader = graph.chain(graph.root, 5, 100);
    const sideline = graph.chain(graph.root, 2, 90);
    if (closed === 'terminal') sideline.at(-1).terminal = { score: 0 };
    else if (closed === 'proven') sideline[0].mateProven = true;
    else sideline.at(-1).children = [];
    assert.deepEqual(graph.choose(), { kind: 'expand', node: leader.at(-1) }, closed);
  }
});

test('catch-up evaluates available leaves without crossing the depth ceiling', () => {
  const graph = tree();
  const leader = graph.chain(graph.root, 5, 100);
  const sideline = graph.add(graph.root, 90);
  const reply = graph.add(sideline, 90, false);
  assert.deepEqual(graph.choose(2), { kind: 'evaluate', node: reply });
  reply.trueScore = reply.value = 90;
  sideline.best = reply;
  assert.equal(graph.choose(2), null);
  assert.deepEqual(graph.choose(3), { kind: 'expand', node: reply });
  assert.equal(leader.at(-1).depth, 5);
});

test('suppression removes catch-up priority without discarding ordinary work', () => {
  const graph = tree();
  const leader = graph.chain(graph.root, 3, 100);
  const lower = graph.chain(graph.root, 7, 90);
  leader.at(-1).children = [];
  lower.at(-1).children = [];
  const third = graph.chain(graph.root, 4, 80);
  const candidate = graph.add(third.at(-1), 80, false);
  assert.deepEqual(graph.choose(), { kind: 'evaluate', node: candidate });
});
