// Scores are White-relative; displayed depth rows retain parent-first ordering.
// This threshold now describes root contenders, not a quota at every depth.
export const DYNAMIC_DEPTH_THRESHOLD = 3;
export const INITIAL_REPLY_COVERAGE = 2;
const EXPLORATION_INTERVAL = 8;
const CLOSE_SCORE = 150;
const SCORE_SCALE = 200;
const RANK_BIAS = 0.25;
const CORRECTION_SCALE = 400;

const scoreOf = node => node.trueScore === null ? node.candidateScore : node.value;
const visitsOf = node => Number.isFinite(node.visits) ? node.visits : 0;
const resolved = node => Boolean(node.terminal || node.mateProven);

function describeDepth(depth, ranked) {
  const firstCandidate = ranked.findIndex(node => node.trueScore === null);
  return { depth, ranked, searchedMoves: firstCandidate < 0 ? Infinity : firstCandidate,
    candidate: firstCandidate < 0 ? null : ranked[firstCandidate] };
}

export function rankDepths(levels, rootSign = 1) {
  const ranksByDepth = [];
  return levels.flatMap((nodes, depth) => {
    if (!depth || !nodes?.length) return [];
    const side = depth % 2 ? rootSign : -rootSign;
    const parentRank = node => ranksByDepth[depth - 1]?.get(node.parent) ?? Infinity;
    const ranked = nodes.slice().sort((a, b) => parentRank(a) - parentRank(b)
      || side * (scoreOf(b) - scoreOf(a)) || a.index - b.index);
    ranksByDepth[depth] = new Map(ranked.map((node, index) => [node, index]));
    return [describeDepth(depth, ranked)];
  });
}

function searchTree(rankings, root, rootSign) {
  // Small standalone callers can omit root. Real search always passes it, so
  // pending root generation remains visible even before a depth row exists.
  const childrenByParent = new Map();
  if (!root) {
    const first = rankings.toSorted((a, b) => a.depth - b.depth).find(level => level.ranked.length);
    root = first?.ranked[0]?.parent ?? (first && { depth: first.depth - 1, children: first.ranked });
    for (const level of rankings) for (const node of level.ranked) {
      if (!node.parent) continue;
      if (!childrenByParent.has(node.parent)) childrenByParent.set(node.parent, []);
      childrenByParent.get(node.parent).push(node);
    }
  }
  const childCache = new Map();
  function children(node) {
    if (!childCache.has(node)) {
      const list = childrenByParent.get(node) ?? node.children ?? [];
      const side = node.depth % 2 ? -rootSign : rootSign;
      childCache.set(node, list.slice().sort((a, b) => side * (scoreOf(b) - scoreOf(a)) || a.index - b.index));
    }
    return childCache.get(node);
  }
  return { root, children };
}

// A static-to-search correction is not statistical uncertainty. Decay its
// scheduling bonus as the branch receives work, so an old refutation cannot
// keep demanding attention forever. Only an unresolved check at the PV tip
// retains forcing priority once continuations have been searched.
const correctionOf = node => node.trueScore === null ? 0
  : Math.abs(node.value - node.trueScore) / Math.sqrt(visitsOf(node) + 1);
const forcingTip = tip => !resolved(tip) && Boolean(tip.forcing);
const competitive = (node, tip, gap) => gap <= CLOSE_SCORE || forcingTip(tip) || correctionOf(node) >= CLOSE_SCORE;

function lineTip(node) {
  const seen = new Set();
  while (node.best && !seen.has(node)) {
    seen.add(node);
    node = node.best;
  }
  return node;
}

export function canDeepen(rankings, maxDepth, { root, rootSign = 1 } = {}) {
  if (maxDepth >= 64) return false;
  const tree = searchTree(rankings, root, rootSign);
  if (!tree.root) return false;
  const contenders = tree.children(tree.root).slice(0, DYNAMIC_DEPTH_THRESHOLD);
  if (!contenders.length || contenders.some(node => node.trueScore === null)) return false;
  if (contenders.length < DYNAMIC_DEPTH_THRESHOLD && tree.root.canWiden && !tree.root.generationDone) return false;
  // A closed selective branch is not terminal proof. It simply has no work
  // that can reach this ceiling. Check the whole branch, since a closed PV tip
  // may still have a searchable sibling.
  function hasWork(node) {
    if (resolved(node)) return false;
    if (node.trueScore === null) return true;
    if (node.depth >= maxDepth) return false;
    if (node.children === null || node.canWiden && !node.generationDone) return true;
    return tree.children(node).some(hasWork);
  }
  function hasLiveCeiling(node) {
    if (resolved(node) || node.trueScore === null) return false;
    if (node.depth >= maxDepth && node.children === null) return true;
    if (node.depth >= maxDepth && node.canWiden && !node.generationDone) return true;
    return tree.children(node).some(hasLiveCeiling);
  }
  const bestScore = scoreOf(contenders[0]) * rootSign;
  let liveCeiling = false;
  for (const contender of contenders) {
    if (resolved(contender)) continue;
    if (maxDepth > contender.depth) {
      const replies = tree.children(contender);
      if (!replies.length) {
        if (hasWork(contender)) return false;
        continue;
      }
      if (replies.slice(0, INITIAL_REPLY_COVERAGE).some(node => node.trueScore === null)) return false;
      // A temporarily short generator is not yet evidence of reply coverage.
      if (replies.length < INITIAL_REPLY_COVERAGE && contender.canWiden) return false;
    }
    const tip = lineTip(contender);
    const gap = Math.max(0, bestScore - scoreOf(contender) * rootSign);
    if (!competitive(contender, tip, gap)) continue;
    // A long single-reply PV is not defensive coverage. Before raising the
    // ceiling, compare the leading replies along that line, including nodes
    // whose narrow tactical extension has become ordinary search.
    for (let current = contender; current && current.depth < maxDepth && !resolved(current); current = current.best) {
      if (current.children === null) return false;
      const replies = tree.children(current);
      if (replies.slice(0, INITIAL_REPLY_COVERAGE).some(node => node.trueScore === null)
        || replies.length < INITIAL_REPLY_COVERAGE && current.canWiden && !current.generationDone) return false;
    }
    if (tip.trueScore === null) return false;
    if (tip.depth < maxDepth && !resolved(tip) && hasWork(contender)) return false;
    liveCeiling ||= hasLiveCeiling(contender);
  }
  // Resolved lines need no deeper ceiling; at least one live continuation must
  // actually reach it. Breadth elsewhere does not hold up a forcing line.
  return liveCeiling;
}

export function chooseWork(rankings, { maxDepth, root, rootSign = 1, canExpand, canEvaluate } = {}) {
  const tree = searchTree(rankings, root, rootSign);
  root = tree.root;
  if (!root) return null;
  const expandable = node => node.depth < maxDepth || Boolean(canExpand?.(node));
  const evaluable = node => !canEvaluate || canEvaluate(node);
  const widenable = node => expandable(node) && node.canWiden && !node.generationDone && !resolved(node);
  const contenders = tree.children(root).slice(0, DYNAMIC_DEPTH_THRESHOLD);

  // Compare a few plausible root moves before spending the budget on one PV.
  // Then search two currently strongest opponent replies for each contender.
  // This coverage is branch-local and remains valid after scores reorder.
  for (const contender of contenders) {
    if (contender.trueScore === null && evaluable(contender)) return { kind: 'evaluate', node: contender };
  }
  if (contenders.length < DYNAMIC_DEPTH_THRESHOLD && widenable(root)) return { kind: 'widen', node: root };
  // Mandatory coverage must not consume every exploration opportunity when
  // score changes repeatedly promote new contenders.
  const rootExplore = visitsOf(root) > 0 && visitsOf(root) % EXPLORATION_INTERVAL === 0;
  for (const contender of rootExplore ? [] : contenders) {
    if (resolved(contender) || !expandable(contender) || contender.trueScore === null) continue;
    if (contender.children === null) return { kind: 'expand', node: contender };
    const replies = tree.children(contender);
    for (const reply of replies.slice(0, INITIAL_REPLY_COVERAGE)) {
      if (reply.trueScore === null && evaluable(reply)) return { kind: 'evaluate', node: reply };
    }
    if (replies.length < INITIAL_REPLY_COVERAGE && widenable(contender)) return { kind: 'widen', node: contender };
  }

  // Allocate exploration and catch-up from each position's own work count.
  // A global modulo can repeatedly miss a branch's exploration turn when
  // unrelated root branches consume those operations.
  function select(node, allowCatchUp = true) {
    let result = null;
    if (node !== root && node.trueScore === null) {
      if (evaluable(node)) result = { kind: 'evaluate', node };
    } else if (!resolved(node) && expandable(node)) {
      if (node.children === null) result = { kind: 'expand', node };
      else {
        const children = tree.children(node);
        const workCount = visitsOf(node);
        const explore = workCount > 0 && workCount % EXPLORATION_INTERVAL === 0;
        const catchUp = !explore && workCount % 4 === 2;
        // Width grows approximately with sqrt(work): 8 -> 16 after 24
        // operations, 32 after 120, 64 after 504. Deepening gets time between
        // batches, while an exhausted frontier widens immediately (depth 1 too).
        const initialWidth = node.initialWidth ?? 8;
        const desiredWidth = initialWidth * Math.sqrt(1 + workCount / initialWidth);
        const pendingReply = !explore && node !== root
          ? children.slice(0, INITIAL_REPLY_COVERAGE).find(child => child.trueScore === null && evaluable(child)) : null;
        if (pendingReply) result = { kind: 'evaluate', node: pendingReply };
        else if (widenable(node) && (children.length < INITIAL_REPLY_COVERAGE
          || children.length * 2 <= desiredWidth)) result = { kind: 'widen', node };
        else {
          const side = node.depth % 2 ? -rootSign : rootSign;
          const bestScore = children.length ? scoreOf(children[0]) * side : 0;
          let shortestAbove = Infinity;
          const available = [];
          for (const [rank, child] of children.entries()) {
            const tip = lineTip(child), length = tip.depth;
            // Preserve suppression throughout demoted branches: a branch
            // already deeper than a stronger alternative has no catch-up claim.
            const suppressed = !allowCatchUp || length > shortestAbove;
            const gap = Math.max(0, bestScore - scoreOf(child) * side);
            const deficit = rank > 0 && !resolved(tip) && (shortestAbove - length >= 2
              || shortestAbove >= maxDepth && length < maxDepth);
            available.push({ child, rank, suppressed,
              catchUp: catchUp && !suppressed && competitive(child, tip, gap) && deficit,
              // A stable rank prior keeps tied evaluations selective: a
              // visit bonus alone degenerates into breadth-first search in
              // large equal-score trees. Explicit exploration supplies fairness.
              priority: -gap / SCORE_SCALE - rank * RANK_BIAS + 0.1 / Math.sqrt(visitsOf(child) + 1)
                + (child.trueScore === null ? 0.1 : 0) + (forcingTip(tip) ? 0.35 : 0)
                + Math.min(CORRECTION_SCALE, correctionOf(child)) / CORRECTION_SCALE,
            });
            shortestAbove = Math.min(shortestAbove, length);
          }
          available.sort((a, b) => explore
            ? visitsOf(a.child) - visitsOf(b.child) || a.rank - b.rank
            : Number(b.catchUp) - Number(a.catchUp) || b.priority - a.priority || a.rank - b.rank);
          // Descend only until a usable frontier is found. Sorting scores is
          // cheap; recursively probing every competing subtree is not.
          for (const choice of available) {
            result = select(choice.child, !choice.suppressed);
            if (result) break;
          }
          result ??= widenable(node) ? { kind: 'widen', node } : null;
        }
      }
    }
    return result;
  }
  return select(root);
}
