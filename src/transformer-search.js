import { createLegalTurnProbe, createPositionKeyCache, formatAction, generateActions, inCheck } from './rules.js';
import { createCandidateStream } from './transformer-candidates.js';
import { createValueCache } from './transformer-value-cache.js';
import { canDeepen, chooseWork, DYNAMIC_DEPTH_THRESHOLD, rankDepths } from './transformer-frontier.js';

export const MATE_SCORE = 100_000;
const MATE_THRESHOLD = MATE_SCORE - 1000;
const sign = position => position.action % 2 === 0 ? 1 : -1;
const finite = (value, fallback, min, max) => Number.isFinite(Number(value))
  ? Math.max(min, Math.min(max, Number(value))) : fallback;
class SearchInterrupted extends Error {}
const PROGRESS_INTERVAL_MS = 100;
const DISPLAY_RANK_LIMIT = 10;

/**
 * Neural value search over complete legal submissions. Distinct partial-move
 * successors are scored in batches to assemble the strongest components first,
 * before progressively widening the full-turn candidate set. Completed candidates
 * use their submitted-state values. A branch-local scheduler allocates selective
 * expansion across depths, backing up only evaluated continuations.
 * Candidate caps still make this selective, not exhaustive full-rule minimax.
 * Values are always White centipawns.
 * The evaluator is required: no classical evaluation replaces a missing model.
 */
export async function analyze(position, options = {}) {
  if (typeof options.evaluateBatch !== 'function') throw new Error('Transformer search requires evaluateBatch.');
  const started = performance.now();
  const timeMs = finite(options.timeMs, 3000, 0, 3_600_000);
  const maxDepth = Math.floor(finite(options.maxDepth, 4, 0, 64));
  const dynamicDepth = maxDepth === 0;
  let currentMaxDepth = dynamicDepth ? 1 : maxDepth;
  const maxNodes = Math.floor(finite(options.maxNodes, 200_000, 0, 1_000_000_000));
  const candidateLimit = Math.floor(finite(options.candidateLimit, 64, 1, 256));
  // The same position must see the same candidate set when it becomes the
  // root. Advanced callers may still opt into a different reply cap, accepting
  // that asymmetry even though both caps now follow neural component ordering.
  const innerCandidateLimit = Math.floor(finite(options.innerCandidateLimit, candidateLimit, 1, 256));
  const deadline = options.unlimitedTime === true ? Infinity : started + timeMs, rootSign = sign(position);
  const keyPosition = createPositionKeyCache();
  const initialCandidates = Math.floor(finite(options.initialCandidates, 8, 1, 64));
  const componentBatchSize = Math.floor(finite(options.componentBatchSize, 16, 1, 128));
  const candidateBranches = Math.floor(finite(options.candidateBranches, 4, 1, 8));
  const extensionDepth = Math.floor(finite(options.tacticalExtensionDepth ?? options.quiescenceDepth, 2, 0, 4));
  const extensionCandidateLimit = Math.floor(finite(options.extensionCandidateLimit, 4, 1, 32));
  const maxValueCacheEntries = Math.floor(finite(options.maxValueCacheEntries, 4096, 0, 131072));
  const valueCacheMemoryMb = finite(options.valueCacheMemoryMb, 32, 0, 1024);
  // Cache lifetime is exactly one analysis, and therefore one fixed model/encoding.
  const terminalCache = new WeakMap();
  const terminalProbes = new Map();
  const deferredEvaluations = new Set();
  const selectedProbePositions = new WeakSet();
  const terminalProbeWork = Math.floor(finite(options.terminalProbeWork, 256, 1, 4096));
  let terminalProbeResumes = 0;
  const valueCache = createValueCache(keyPosition, { maxEntries: maxValueCacheEntries, maxBytes: valueCacheMemoryMb * 1024 * 1024 });
  const openStreams = new Set();
  const timings = { generationMs: 0, inferenceMs: 0, policyMs: 0, terminalMs: 0, schedulingMs: 0 };
  let wideningSteps = 0, policyCalls = 0, qnodes = 0, effectiveQuiescenceDepth = 0;
  const levels = [];
  const root = { position, depth: 0, children: null, parent: null, visits: 0, canWiden: false, generationDone: false };
  let nextIndex = 0, trueEvaluations = 0;
  let nodes = 0, searchNodes = 0, generationNodes = 0, evaluations = 0, inferenceBatches = 0;
  let depth = 0, searchingDepth = 0, selectiveDepth = 0, rootActionsSearched = 0;
  let bestAction = null, pv = [], score = null, rootFallback = null;
  let completed = false, status = 'incomplete', stoppedReason = null;
  let mateProven = false;
  let candidateCaps = 0, lastProgress = started;
  const notationCache = new WeakMap();

  function check(includeNodes = true) {
    if (options.shouldStop?.()) stoppedReason = 'cancelled';
    else if (includeNodes && nodes >= maxNodes) stoppedReason = 'nodes';
    else if (performance.now() >= deadline) stoppedReason = 'time';
    else return;
    throw new SearchInterrupted();
  }
  function tick(kind = 'generation') {
    check(); nodes++;
    if (kind === 'generation') generationNodes++;
    else searchNodes++;
    reportProgress();
  }
  function reportProgress(force = false) {
    if (!options.onProgress || bestAction === null) return;
    const now = performance.now();
    if (!force && now - lastProgress < PROGRESS_INTERVAL_MS) return;
    lastProgress = now;
    options.onProgress(snapshot());
  }
  function retainRootCandidate(moves, next, terminal = null) {
    if (completed) return;
    const value = terminal ? -terminal.score : valueCache.get(next) * rootSign;
    if (Number.isFinite(value) && (!rootFallback || value > rootFallback.score)) {
      rootFallback = { bestAction: moves, pv: [moves], score: value, mateProven: Boolean(terminal?.mateProven) };
    }
  }
  function terminalValue(pos, ply) {
    return { score: inCheck(pos) ? -MATE_SCORE + ply : 0, pv: [], terminal: true, mateProven: true };
  }
  // A single legal submission disproves terminal status. Only reaching done
  // without a work/time interruption proves mate or stalemate.
  function probeTerminal(pos, ply, maxWork = Infinity) {
    const probeStarted = performance.now();
    try {
      if (!terminalCache.has(pos)) {
        tick('search');
        selectiveDepth = Math.max(selectiveDepth, ply);
        let probe = terminalProbes.get(pos);
        if (!probe) {
          // Keep suspended traversal memory bounded even when construction
          // touches many difficult states that are never selected for search.
          if (terminalProbes.size >= 128) {
            const oldest = [...terminalProbes.keys()].find(state => !selectedProbePositions.has(state));
            terminalProbes.get(oldest).close(); terminalProbes.delete(oldest);
          }
          probe = createLegalTurnProbe(pos, { tick, keyPosition, skipOptionalSpatial: false });
        } else { terminalProbeResumes++; terminalProbes.delete(pos); }
        terminalProbes.set(pos, probe);
        const result = probe.advance(maxWork);
        if (result.done) {
          terminalCache.set(pos, result.terminal); terminalProbes.delete(pos); selectedProbePositions.delete(pos);
        }
      }
      return terminalCache.get(pos) ? terminalValue(pos, ply) : null;
    } finally { timings.terminalMs += performance.now() - probeStarted; }
  }
  async function infer(positions) {
    // Reuse exact-history values across separately constructed branches as well
    // as duplicate states within the same batch.
    const missing = new Map();
    for (const pos of positions) if (!valueCache.has(pos)) {
      const key = pos.action + ':' + keyPosition(pos);
      if (!missing.has(key)) missing.set(key, []);
      missing.get(key).push(pos);
    }
    if (!missing.size) return;
    const groups = [...missing.values()];
    positions = groups.map(group => group[0]);
    const inferenceStarted = performance.now();
    for (const unused of positions) tick('search');
    check(false);
    // Waiting for a model must remain cancellable even if its transport stalls.
    // The evaluator may finish later; Promise.race consumes its rejection and
    // no late result can update this search's caches or principal variation.
    let timer;
    const interrupted = new Promise((resolve, reject) => {
      const poll = () => {
        try { check(false); reportProgress(); }
        catch (error) { reject(error); return; }
        timer = setTimeout(poll, Math.min(25, Math.max(1, deadline - performance.now())));
      };
      timer = setTimeout(poll, Math.min(25, Math.max(1, deadline - performance.now())));
    });
    let values;
    try {
      inferenceBatches++;
      values = await Promise.race([Promise.resolve().then(() => options.evaluateBatch(positions)), interrupted]);
    } finally { clearTimeout(timer); timings.inferenceMs += performance.now() - inferenceStarted; }
    check(false);
    if (!Array.isArray(values) || values.length !== positions.length || values.some(value => typeof value !== 'number' || !Number.isFinite(value))) {
      throw new Error('Transformer evaluator must return one finite White-centipawn number per position.');
    }
    evaluations += values.length;
    groups.forEach((group, index) => {
      const value = Math.max(-MATE_THRESHOLD + 1, Math.min(MATE_THRESHOLD - 1, values[index]));
      for (const pos of group) valueCache.set(pos, value);
    });
  }
  async function scoreMoves(pos, moves) {
    if (typeof options.scoreMoves !== 'function') return null;
    check();
    const policyStarted = performance.now();
    let timer;
    const interrupted = new Promise((resolve, reject) => {
      const poll = () => {
        try { check(false); reportProgress(); }
        catch (error) { reject(error); return; }
        timer = setTimeout(poll, 25);
      };
      timer = setTimeout(poll, 25);
    });
    try {
      const scores = await Promise.race([Promise.resolve().then(() => options.scoreMoves(pos, moves)), interrupted]);
      check(false);
      if (scores === null) return null;
      if (!Array.isArray(scores) || scores.length !== moves.length || !scores.every(Number.isFinite)) {
        throw new Error('Component policy must return one finite score per move, or null.');
      }
      policyCalls++;
      return scores;
    } finally { clearTimeout(timer); timings.policyMs += performance.now() - policyStarted; }
  }
  function canExpand(node) {
    if (node.terminal || node.mateProven || node.depth >= 64) return false;
    if (node.depth < currentMaxDepth) return true;
    // inCheck uses full temporal royal threats. Captures and timeline creation
    // alone do not trigger extensions.
    return node.forcing && node.depth < currentMaxDepth + extensionDepth;
  }
  function recordWork(node) {
    for (let current = node; current; current = current.parent) current.visits = (current.visits || 0) + 1;
  }
  async function expand(node) {
    searchingDepth = node.depth + 1;
    const generationStarted = performance.now();
    const nestedBefore = timings.inferenceMs + timings.policyMs + timings.terminalMs;
    const first = node.children === null;
    if (first) {
      node.children = [];
      node.limit = node.depth === 0 ? candidateLimit : innerCandidateLimit;
      node.stream = createCandidateStream(node.position, {
        ply: node.depth, tick, keyPosition, infer, valueFor: pos => valueCache.get(pos),
        probeTerminal, retainRootCandidate, componentBatchSize, scoreMoves, candidateBranches,
      });
      node.canWiden = true;
      openStreams.add(node.stream);
    } else wideningSteps++;
    const allowance = node.depth >= currentMaxDepth ? Math.min(node.limit, extensionCandidateLimit) : node.limit;
    const target = Math.min(allowance, first ? initialCandidates : Math.max(initialCandidates, node.children.length * 2));
    try {
      while (!node.generationDone && node.children.length < target) {
        const next = await node.stream.next();
        if (next.done) {
          node.generationDone = true;
          // Search intentionally omits optional same-board moves. Exhausting
          // that selection cannot certify that every legal reply was examined.
          node.exhaustive = !node.stream.selective;
          break;
        }
        const child = { ...next.value, parent: node, depth: node.depth + 1, index: nextIndex++,
          trueScore: null, value: null, children: null, terminal: null,
          mateProven: false, best: null, visits: 0, canWiden: false, generationDone: false,
        };
        node.children.push(child);
        (levels[child.depth] ??= []).push(child);
        selectiveDepth = Math.max(selectiveDepth, child.depth);
        if (node === root) retainRootCandidate(child.moves, child.position,
          terminalCache.get(child.position) ? terminalValue(child.position, child.depth) : null);
      }
      if (node.children.length >= node.limit && !node.generationDone) {
        node.generationDone = true; node.exhaustive = false; candidateCaps++;
      }
      // The extension allowance is temporary. In dynamic mode this node may
      // later fall inside the ordinary horizon and resume its wider generator.
      node.canWiden = !node.generationDone && node.children.length < allowance;
      if (node.generationDone) {
        await node.stream.return(); openStreams.delete(node.stream); node.stream = null;
      }
      // Admitting unknown alternatives invalidates a previous all-replies loss
      // certificate. Backups always distinguish estimates from proved outcomes.
      backup(node); publishRoot();
    } finally {
      timings.generationMs += Math.max(0, performance.now() - generationStarted
        - (timings.inferenceMs + timings.policyMs + timings.terminalMs - nestedBefore));
    }
  }
  function backup(node) {
    for (let current = node; current; current = current.parent) {
      const side = sign(current.position);
      const evaluated = current.children?.filter(child => child.trueScore !== null) ?? [];
      const best = evaluated.reduce((best, child) =>
        !best || child.value * side > best.value * side ? child : best, null);
      current.best = best;
      if (!best) {
        current.value = current.trueScore;
        current.mateProven = Boolean(current.terminal?.mateProven);
        continue;
      }
      current.value = best.value;
      current.mateProven = false;
      if (Math.abs(current.value) > MATE_THRESHOLD) {
        current.mateProven = current.value * side > 0 ? best.mateProven
          : current.exhaustive && evaluated.length === current.children.length
            && evaluated.every(child => child.value * side < -MATE_THRESHOLD && child.mateProven);
        if (!current.mateProven) current.value = Math.sign(current.value) * (MATE_THRESHOLD - 1);
      }
    }
  }
  function publishRoot() {
    if (!root.best) return;
    score = root.value * rootSign;
    mateProven = Math.abs(score) > MATE_THRESHOLD && root.mateProven;
    pv = [];
    for (let node = root.best; node; node = node.best) pv.push(node.moves);
    bestAction = pv[0];
    completed = true;
    status = 'ok';
  }
  async function evaluateCandidate(node) {
    searchingDepth = node.depth;
    tick('search');
    selectedProbePositions.add(node.position);
    const terminal = probeTerminal(node.position, node.depth, terminalProbeWork);
    if (terminalCache.has(node.position)) selectedProbePositions.delete(node.position);
    if (!terminalCache.has(node.position)) {
      deferredEvaluations.add(node);
      return;
    }
    deferredEvaluations.delete(node);
    // The final component often already evaluated this exact submitted state.
    // Reusing it avoids a duplicate model call without changing scheduling.
    if (!terminal && !valueCache.has(node.position)) await infer([node.position]);
    check(false);
    node.terminal = terminal;
    node.forcing = !terminal && inCheck(node.position);
    node.trueScore = terminal ? terminal.score * sign(node.position) : valueCache.get(node.position);
    if (node.depth > currentMaxDepth) {
      qnodes++; effectiveQuiescenceDepth = Math.max(effectiveQuiescenceDepth, node.depth - currentMaxDepth);
    }
    node.value = node.trueScore;
    node.mateProven = Boolean(terminal?.mateProven);
    trueEvaluations++;
    if (node.depth === 1) rootActionsSearched++;
    const previousDepth = depth;
    depth = Math.max(depth, node.depth);
    backup(node);
    publishRoot();
    reportProgress(depth > previousDepth);
  }
  function notationFor(node) {
    if (!notationCache.has(node)) {
      notationCache.set(node, formatAction(node.parent.position, node.moves) || 'Submit turn');
    }
    return notationCache.get(node);
  }
  function displayEntry(node, index) {
    const isTrue = node.trueScore !== null;
    const value = isTrue ? node.value : node.candidateScore;
    const provenMate = isTrue && node.mateProven && Math.abs(value) > MATE_THRESHOLD;
    const line = [];
    // Retain the route from the root: entries at the same depth can belong to
    // different branches, so their move notation alone is not enough context.
    for (let current = node; current.parent; current = current.parent) line.push(notationFor(current));
    line.reverse();
    for (let current = node.best; current; current = current.best) line.push(notationFor(current));
    return {
      id: node.index, rank: index + 1, evaluationType: isTrue ? 'true' : 'candidate',
      score: Math.round(value), scoreType: provenMate ? 'mate' : 'cp',
      mateIn: provenMate ? Math.sign(value) * (MATE_SCORE - Math.abs(value)) : null,
      notation: notationFor(node), line, expanded: node.children !== null,
      staticScore: isTrue ? Math.round(node.trueScore) : Math.round(node.candidateScore),
      searchedReplies: node.children?.filter(child => child.trueScore !== null).length ?? 0,
      generatedReplies: node.children?.length ?? 0,
      repliesExhaustive: Boolean(node.exhaustive), forcing: Boolean(node.forcing), visits: node.visits,
    };
  }
  function snapshot() {
    const elapsedMs = Math.max(0, performance.now() - started);
    const whiteScore = score === null ? null : Math.round(score * rootSign);
    const rankings = rankDepths(levels, rootSign);
    const commonPrefix = Math.min(...rankings.map(level => level.searchedMoves));
    return {
      engine: 'transformer', bestAction, score: whiteScore, depth, nodes, searchNodes, generationNodes,
      qnodes, ttHits: valueCache.hits, qTtHits: 0, cutoffs: 0, elapsedMs: Math.round(elapsedMs),
      searchingDepth, rootActionsSearched, selectiveDepth,
      depthMode: dynamicDepth ? 'dynamic' : 'fixed', currentMaxDepth,
      dynamicDepthThreshold: dynamicDepth ? DYNAMIC_DEPTH_THRESHOLD : null,
      nps: elapsedMs ? Math.round(nodes * 1000 / elapsedMs) : 0, pv, status, completed,
      stoppedReason, tableEntries: valueCache.size, cacheMemoryBytes: valueCache.bytes,
      searchPolicy: 'transformer-adaptive-depth', candidateLimit, innerCandidateLimit,
      trueEvaluations, pvDepth: pv.length,
      progressIntervalMs: PROGRESS_INTERVAL_MS,
      expansionRank: Number.isFinite(commonPrefix) ? commonPrefix : null,
      rankings: rankings.map(level => ({
        depth: level.depth, side: (level.depth % 2 ? rootSign : -rootSign) > 0 ? 'white' : 'black',
        total: level.ranked.length,
        searchedMoves: Number.isFinite(level.searchedMoves) ? level.searchedMoves : null,
        entries: level.ranked.slice(0, DISPLAY_RANK_LIMIT).map(displayEntry),
      })),
      depthStats: rankings.map(level => ({
        depth: level.depth, candidates: level.ranked.filter(node => node.trueScore === null).length,
        trueEvaluations: level.ranked.filter(node => node.trueScore !== null).length,
        searchedMoves: Number.isFinite(level.searchedMoves) ? level.searchedMoves : null,
        topCandidateRank: level.candidate ? level.searchedMoves + 1 : null,
      })),
      candidateCaps, candidateCacheEntries: 0, evaluations, inferenceBatches, policyLeaves: 0, effectiveQuiescenceDepth,
      wideningSteps, policyCalls, valueCacheHits: valueCache.hits,
      terminalProbeResumes, pendingTerminalProbes: terminalProbes.size,
      ...(options.collectPolicyAlternatives ? { policyAlternatives: (root.children ?? [])
        .filter(node => node.trueScore !== null && Number.isFinite(node.value))
        .map(node => ({ action: node.moves, score: node.value })) } : {}),
      timings: Object.fromEntries(Object.entries(timings).map(([key, value]) => [key, Math.round(value)])),
      optionalMovePolicy: 'temporal-only',
      mateProven, terminalProof: ['checkmate', 'stalemate'].includes(status) ? 'unrestricted-legal-exhaustion' : null,
      scoreType: score === null ? 'unavailable' : Math.abs(score) > MATE_THRESHOLD ? 'mate' : 'cp',
      mateIn: score !== null && Math.abs(score) > MATE_THRESHOLD ? Math.sign(whiteScore) * (MATE_SCORE - Math.abs(score)) : null,
      limits: { timeMs, maxDepth, maxNodes, candidateLimit, innerCandidateLimit, initialCandidates, componentBatchSize, quiescenceDepth: extensionDepth,
        tacticalExtensionDepth: extensionDepth, extensionCandidateLimit, maxValueCacheEntries, valueCacheMemoryMb, terminalProbeWork, candidateBranches },
    };
  }
  try {
    // The playable fallback obeys the same optional-board policy as search.
    // Exhausting this restricted traversal never certifies a terminal root.
    const legal = generateActions(position, { tick, keyPosition, skipOptionalSpatial: true, firstOnly: true });
    try {
      const first = legal.next();
      if (!first.done) { terminalCache.set(position, false); bestAction = first.value.moves; pv = [bestAction]; }
    } finally { legal.return?.(); }
    if (bestAction === null) probeTerminal(position, 0);
    if (!terminalCache.get(position)) await expand(root);
    else root.children = [];
    if (terminalCache.get(position)) {
      const result = terminalValue(position, 0);
      score = result.score; mateProven = Math.abs(score) > MATE_THRESHOLD && result.mateProven; completed = true;
      bestAction = null; pv = [];
      status = score ? 'checkmate' : 'stalemate';
      stoppedReason = 'terminal';
    } else if (!root.children.length) {
      status = 'policy-limited'; stoppedReason = 'policy';
    } else {
      while (true) {
        check();
        const schedulingStarted = performance.now();
        const rankings = rankDepths(levels, rootSign);
        if (dynamicDepth && canDeepen(rankings, currentMaxDepth, { root, rootSign })) {
          currentMaxDepth++;
          for (const level of levels) for (const node of level ?? []) {
            if (node.stream && !node.generationDone) {
              const allowance = node.depth >= currentMaxDepth ? Math.min(node.limit, extensionCandidateLimit) : node.limit;
              node.canWiden = node.children.length < allowance;
            }
          }
          reportProgress(true);
          check();
        }
        let work;
        // Give deferred proofs one operation in four, in FIFO order. Other
        // contenders remain eligible while an expensive proof is unfinished.
        if (!deferredEvaluations.size || root.visits % 4 !== 3) {
          work = chooseWork(rankings, { maxDepth: currentMaxDepth, root, rootSign, canExpand,
            canEvaluate: node => !deferredEvaluations.has(node)
              && (deferredEvaluations.size < 32 || terminalCache.has(node.position)) });
        }
        if (!work && deferredEvaluations.size) {
          const node = deferredEvaluations.values().next().value;
          deferredEvaluations.delete(node);
          work = { kind: 'evaluate', node };
        }
        timings.schedulingMs += performance.now() - schedulingStarted;
        if (!work) { stoppedReason = depth >= currentMaxDepth ? 'depth' : 'frontier'; break; }
        if (work.kind === 'expand' || work.kind === 'widen') await expand(work.node);
        else await evaluateCandidate(work.node);
        recordWork(work.node);
        check(false);
        if (mateProven) { stoppedReason = 'mate'; break; }
      }
    }
  } catch (error) {
    if (!(error instanceof SearchInterrupted)) throw error;
    if (!completed && rootFallback) {
      bestAction = rootFallback.bestAction; pv = rootFallback.pv;
      // A partial root cannot certify a loss over unsearched alternatives.
      score = rootFallback.score < -MATE_THRESHOLD ? -MATE_THRESHOLD + 1 : rootFallback.score;
      mateProven = Math.abs(score) > MATE_THRESHOLD && Boolean(rootFallback.mateProven);
    }
  } finally {
    try { for (const stream of openStreams) await stream.return(); }
    finally {
      for (const probe of terminalProbes.values()) probe.close();
      terminalProbes.clear();
    }
  }
  return snapshot();
}
