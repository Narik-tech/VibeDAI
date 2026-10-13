import test from 'node:test';
import assert from 'node:assert/strict';
import { canDeepen, chooseWork, DYNAMIC_DEPTH_THRESHOLD, rankDepths } from '../src/transformer-frontier.js';

const node = (depth, index, candidateScore, trueScore = null, extra = {}) => ({
  depth, index, candidateScore, trueScore, value: trueScore,
  children: null, terminal: null, ...extra,
});
const ranked = (levels, rootSign = 1) => rankDepths(levels, rootSign);

function tree(rootSign = 1) {
  let nextIndex = 0;
  const root = { depth: 0, children: [], best: null, visits: 0 };
  const levels = [[]];
  function add(parent, score, isTrue = true, extra = {}) {
    const value = score * rootSign;
    const child = node(parent.depth + 1, nextIndex++, value, isTrue ? value : null,
      { parent, best: null, visits: 0, ...extra });
    (parent.children ??= []).push(child);
    (levels[child.depth] ??= []).push(child);
    if (isTrue) parent.best ??= child;
    return child;
  }
  const rankings = () => ranked(levels, rootSign);
  return { root, levels, add, rankings,
    choose: (options = {}) => chooseWork(rankings(), { maxDepth: 8, root, rootSign, ...options }),
    ready: maxDepth => canDeepen(rankings(), maxDepth, { root, rootSign }),
  };
}

function complete(work) {
  for (let current = work.node; current; current = current.parent) current.visits++;
  if (work.kind === 'evaluate') {
    work.node.trueScore = work.node.value = work.node.candidateScore;
    work.node.parent.best ??= work.node;
  }
}

test('depth rankings alternate the mover perspective and use backed True values', () => {
  const first = [node(1, 0, 10), node(1, 1, 90), node(1, 2, 1000, 50, { value: 70 })];
  const second = [node(2, 3, 10), node(2, 4, 90), node(2, 5, -1000, 50, { value: 70 })];
  assert.deepEqual(ranked([[], first, second]).map(row => row.ranked),
    [[first[1], first[2], first[0]], [second[0], second[2], second[1]]]);
  assert.deepEqual(ranked([[], first, second], -1).map(row => row.ranked),
    [[first[0], first[2], first[1]], [second[1], second[2], second[0]]]);
});

test('equal values retain insertion order without mutating level arrays', () => {
  const first = node(1, 0, 10), second = node(1, 1, 10, 10), third = node(1, 2, 10);
  const level = [third, first, second];
  assert.deepEqual(ranked([[], level])[0].ranked, [first, second, third]);
  assert.deepEqual(level, [third, first, second]);
});

test('display rankings propagate parent priority after backed values change', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign);
    const first = graph.add(graph.root, 100), second = graph.add(graph.root, 50);
    const reply = graph.add(first, 10), otherReply = graph.add(second, -1000);
    const continuation = graph.add(reply, 20), otherContinuation = graph.add(otherReply, 10000);
    assert.deepEqual(graph.rankings()[1].ranked, [reply, otherReply]);
    assert.deepEqual(graph.rankings()[2].ranked, [continuation, otherContinuation]);
    first.value = 0;
    assert.deepEqual(graph.rankings()[1].ranked, [otherReply, reply]);
    assert.deepEqual(graph.rankings()[2].ranked, [otherContinuation, continuation]);
  }
});

test('Searched Moves remains the displayed True prefix, independent of work allocation', () => {
  const graph = tree();
  const leader = graph.add(graph.root, 90), candidate = graph.add(graph.root, 80, false);
  graph.add(graph.root, 70);
  assert.equal(graph.rankings()[0].searchedMoves, 1);
  leader.value = 50;
  assert.equal(graph.rankings()[0].searchedMoves, 0);
  assert.equal(graph.rankings()[0].candidate, candidate);
  complete({ kind: 'evaluate', node: candidate });
  assert.equal(graph.rankings()[0].searchedMoves, Infinity);
});

test('initial root coverage evaluates three current contenders before expansion', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign);
    const roots = Array.from({ length: 8 }, (_, index) => graph.add(graph.root, 100 - index, false));
    for (const contender of roots.slice(0, 3)) {
      const work = graph.choose();
      assert.deepEqual(work, { kind: 'evaluate', node: contender });
      complete(work);
    }
    assert.deepEqual(graph.choose(), { kind: 'expand', node: roots[0] });
    assert.equal(roots[3].trueScore, null, 'remaining root breadth does not block reply search');
  }
});

test('two strongest replies to every contender precede a deeper principal variation', () => {
  const graph = tree();
  const roots = [100, 90, 80].map(score => graph.add(graph.root, score));
  const expected = [];
  for (const parent of roots) {
    const replies = [-100, 0, 100].map(score => graph.add(parent, score, false));
    expected.push(...replies.slice(0, 2));
  }
  for (const reply of expected) {
    const work = graph.choose();
    assert.deepEqual(work, { kind: 'evaluate', node: reply });
    complete(work);
  }
  assert.equal(graph.choose().kind, 'expand', 'the scheduler can now deepen an evaluated reply');
});

test('selected internal lines compare leading replies before continuing deeper', () => {
  for (const rootSign of [1, -1]) {
    const graph = tree(rootSign);
    const contender = graph.add(graph.root, 100);
    const reply = graph.add(contender, 100);
    graph.add(contender, 110);
    graph.add(reply, 100);
    const unexplored = graph.add(reply, 90, false);
    assert.equal(graph.ready(3), false, 'a long single-reply PV lacks defensive coverage');
    assert.deepEqual(graph.choose(), { kind: 'evaluate', node: unexplored });
    complete({ kind: 'evaluate', node: unexplored });
    assert.equal(graph.ready(3), true);
  }
});

test('root contender membership responds immediately to backed score changes', () => {
  const graph = tree();
  const roots = [100, 90, 80, 70].map(score => graph.add(graph.root, score));
  for (const parent of roots.slice(0, 3)) graph.add(parent, 0, true, { children: [] });
  roots[0].value = -1000;
  assert.deepEqual(graph.choose(), { kind: 'expand', node: roots[3] });
});

test('progressive widening waits between batches and widens an exhausted root', () => {
  const graph = tree();
  graph.root.canWiden = true;
  const roots = Array.from({ length: 8 }, (_, index) => graph.add(graph.root, 100 - index));
  for (const parent of roots.slice(0, 3)) graph.add(parent, 0);
  graph.root.visits = 23;
  assert.notEqual(graph.choose().kind, 'widen');
  graph.root.visits = 24;
  assert.deepEqual(graph.choose(), { kind: 'widen', node: graph.root });
  graph.root.visits = 1;
  assert.deepEqual(graph.choose({ maxDepth: 1 }), { kind: 'widen', node: graph.root });
  graph.root.generationDone = true;
  assert.equal(graph.choose({ maxDepth: 1 }), null);
});

test('fixed depth one evaluates later widening batches to exhaustion', () => {
  const graph = tree();
  graph.root.canWiden = true;
  for (let index = 0; index < 8; index++) graph.add(graph.root, -index, false);
  let evaluated = 0, widened = 0;
  for (let steps = 0; steps < 100; steps++) {
    const work = graph.choose({ maxDepth: 1 });
    if (!work) break;
    complete(work);
    if (work.kind === 'evaluate') evaluated++;
    else {
      assert.equal(work.kind, 'widen');
      widened++;
      for (let index = 8; index < 16; index++) graph.add(graph.root, -index, false);
      graph.root.canWiden = false;
      graph.root.generationDone = true;
    }
  }
  assert.equal(evaluated, 16);
  assert.equal(widened, 1);
  assert.equal(graph.choose({ maxDepth: 1 }), null);
});

test('periodic exploration reaches very weak branches and finite work completes', () => {
  const graph = tree();
  const roots = [0, -5000, -10000, -15000].map(score => graph.add(graph.root, score, false));
  let fourthVisitedAt = null, evaluations = 0, expansions = 0, steps = 0;
  for (; steps < 500; steps++) {
    const work = graph.choose({ maxDepth: 3 });
    if (!work) break;
    complete(work);
    if (work.kind === 'evaluate') {
      evaluations++;
      if (work.node === roots[3]) fourthVisitedAt = steps;
    } else {
      assert.equal(work.kind, 'expand');
      expansions++;
      for (let index = 0; index < 4; index++) graph.add(work.node, work.node.candidateScore - index, false);
    }
  }
  assert.ok(fourthVisitedAt < 25, `weak root move was evaluated at operation ${fourthVisitedAt}`);
  assert.equal(evaluations, 4 + 16 + 64);
  assert.equal(expansions, 4 + 16);
  assert.ok(steps < 500);
});

test('equal-score branching eight reaches depth eight with selective work', () => {
  const graph = tree();
  for (let index = 0; index < 8; index++) graph.add(graph.root, 0, false);
  let deepestEvaluation = 0, steps = 0;
  // Include two-reply coverage along selected internal lines as well as the
  // root contenders. This remains far below the full eight-way depth-8 tree.
  const workLimit = 128;
  for (; steps < workLimit && deepestEvaluation < 8; steps++) {
    const work = graph.choose({ maxDepth: 8 });
    assert.ok(work);
    complete(work);
    if (work.kind === 'evaluate') deepestEvaluation = Math.max(deepestEvaluation, work.node.depth);
    else for (let index = 0; index < 8; index++) graph.add(work.node, 0, false);
  }
  assert.equal(deepestEvaluation, 8,
    'equal neural scores must not turn the entire eight-way tree into breadth-first search');
  assert.ok(steps < workLimit);
  assert.ok(graph.root.children.slice(0, 3).every(child => child.children?.filter(reply => reply.trueScore !== null).length >= 2),
    'selective depth retains initial defensive coverage under three root contenders');
});

test('forcing callback permits bounded extensions and optional evaluation gating', () => {
  const graph = tree();
  const first = graph.add(graph.root, 0, true, { forcing: true });
  assert.equal(graph.choose({ maxDepth: 1 }), null);
  const canExpand = node => node.forcing && node.depth < 3;
  assert.deepEqual(graph.choose({ maxDepth: 1, canExpand }), { kind: 'expand', node: first });
  const reply = graph.add(first, 0, false, { forcing: true });
  assert.deepEqual(graph.choose({ maxDepth: 1, canExpand }), { kind: 'evaluate', node: reply });
  assert.equal(graph.choose({ maxDepth: 1, canExpand, canEvaluate: () => false }), null);
  complete({ kind: 'evaluate', node: reply });
  assert.deepEqual(graph.choose({ maxDepth: 1, canExpand }), { kind: 'expand', node: reply });
  const leaf = graph.add(reply, 0, true, { forcing: true });
  assert.equal(graph.choose({ maxDepth: 1, canExpand }), null);
  assert.equal(leaf.depth, 3);
});

test('terminal and proven nodes are not expanded or widened', () => {
  const graph = tree();
  graph.add(graph.root, 100, true, { terminal: { score: 100 }, canWiden: true });
  graph.add(graph.root, 90, true, { mateProven: true, canWiden: true });
  assert.equal(graph.choose(), null);
});

test('standalone roots can be derived and empty frontiers finish', () => {
  const first = node(1, 0, 100, 100);
  assert.deepEqual(chooseWork(ranked([[], [first]]), { maxDepth: 2 }), { kind: 'expand', node: first });
  assert.equal(chooseWork([], { maxDepth: 8 }), null);
});

test('dynamic depth opens after three root contenders, not a global breadth quota', () => {
  const graph = tree();
  const roots = Array.from({ length: 8 }, (_, index) => graph.add(graph.root, -index, index < 2));
  assert.equal(DYNAMIC_DEPTH_THRESHOLD, 3);
  assert.equal(graph.ready(1), false);
  complete({ kind: 'evaluate', node: roots[2] });
  assert.equal(graph.ready(1), true);
  assert.equal(roots[3].trueScore, null);
  assert.equal(graph.ready(2), false, 'a new ceiling must earn actual continuation coverage');
});

test('dynamic readiness requires replies under each contender and principal progress', () => {
  const graph = tree();
  const roots = [100, 90, 80].map(score => graph.add(graph.root, score));
  const replies = roots.map(parent => [graph.add(parent, -10), graph.add(parent, 0)]);
  assert.equal(graph.ready(2), true);
  replies[2][1].trueScore = null;
  assert.equal(graph.ready(2), false, 'many replies elsewhere cannot substitute for the third contender');
  replies[2][1].trueScore = 0;
  graph.add(replies[0][0], 10);
  graph.add(replies[1][0], 10);
  assert.equal(graph.ready(3), false);
  graph.add(replies[2][0], 10);
  assert.equal(graph.ready(3), true);
});

test('dynamic readiness permits short resolved lines, but not unfinished reply generation', () => {
  const graph = tree();
  const parent = graph.add(graph.root, 100);
  const reply = graph.add(parent, 0);
  assert.equal(graph.ready(2), true, 'a fully generated one-reply branch is sufficient');
  parent.canWiden = true;
  assert.equal(graph.ready(2), false);
  parent.canWiden = false;
  reply.terminal = { score: 0 };
  assert.equal(graph.ready(2), false, 'entirely resolved lines do not need a deeper ceiling');
  graph.add(graph.root, 90, true, { terminal: { score: 90 } });
  reply.terminal = null;
  assert.equal(graph.ready(2), true);
});

test('dynamic readiness handles reranking, missing frontiers, and depth ceiling 64', () => {
  const graph = tree();
  const roots = [100, 90, 80].map(score => graph.add(graph.root, score));
  const pending = graph.add(graph.root, 70, false);
  assert.equal(graph.ready(1), true);
  roots[0].value = 0;
  assert.equal(graph.ready(1), false);
  complete({ kind: 'evaluate', node: pending });
  assert.equal(graph.ready(1), true);
  assert.equal(graph.ready(2), false);
  assert.equal(graph.ready(64), false);
  assert.equal(canDeepen([], 1), false);
});
