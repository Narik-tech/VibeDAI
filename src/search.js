import { createPositionKeyCache, generateActions, inCheck } from './rules.js';
import { evaluate, pieceValue } from './evaluate.js';
import { normalizeHeuristics } from './heuristics.js';
import { SearchCache } from './search-cache.js';

export const MATE_SCORE = 100_000;
const MATE_THRESHOLD = MATE_SCORE - 1000;
const INF = 1_000_000;
class SearchInterrupted extends Error {}
const actionKey = action => JSON.stringify(action);
const moveKey = move => JSON.stringify(move);
const colorSign = position => position.action % 2 === 0 ? 1 : -1;
const toTable = (score, ply) => score > MATE_THRESHOLD ? score + ply : score < -MATE_THRESHOLD ? score - ply : score;
const fromTable = (score, ply) => score > MATE_THRESHOLD ? score - ply : score < -MATE_THRESHOLD ? score + ply : score;

function finiteOption(value, fallback, min, max) {
  return Number.isFinite(Number(value)) ? Math.max(min, Math.min(max, Number(value))) : fallback;
}

function moveFeatures(position, move, heuristics) {
  const [from, to] = move;
  const mover = position.board[from[0]]?.[from[1]]?.[from[2]]?.[from[3]] || 0;
  const captured = position.board[to[0]]?.[to[1]]?.[to[2]]?.[to[3]] || 0;
  const promotion = to.length > 4 ? pieceValue(to[4], heuristics) - pieceValue(mover, heuristics) : 0;
  // The third coordinate is the captured pawn in an en passant move; castling
  // contains a fourth coordinate and is not a capture.
  const captureValue = pieceValue(captured, heuristics) || (move.length === 3 ? heuristics.pawnValue : 0);
  return { mover, captureValue, promotion, isCapture: !!captured || move.length === 3, isPromotion: to.length > 4,
    temporal: from[0] !== to[0] || from[1] !== to[1] };
}

function historyKey(position, move) {
  const [from, to] = move;
  const piece = Math.abs(position.board[from[0]]?.[from[1]]?.[from[2]]?.[from[3]] || 0);
  // Share learned quiet-move ordering across turns and spatial timelines.
  // Absolute half-turn numbers made the old history disappear every ply.
  // Temporal moves keep their timeline endpoints and relative time distance.
  const lines = from[0] === to[0] ? 's' : `${from[0]},${to[0]}`;
  return `${piece}:${lines}:${to[1] - from[1]}:${from[2]},${from[3]}:${to[2]},${to[3]},${to[4] || 0}`;
}

/**
 * Search complete submitted turns under the requested present-spatial policy:
 * ordinary moves from optional boards are excluded, cross-board moves remain.
 * Alpha-beta bounds apply to that selective tree, not every legal action.
 * `nodes` includes generator work ticks, making maxNodes deterministic even
 * when finding one legal multiboard action takes considerable work.
 */
export function createSearchSession(position, options = {}) {
  const started = performance.now();
  const configuredHeuristics = normalizeHeuristics(options.heuristics);
  const qDepth = options.quiescenceDepth === undefined ? configuredHeuristics.quiescenceDepth
    : Math.floor(finiteOption(options.quiescenceDepth, 2, 0, 8));
  const heuristics = qDepth === configuredHeuristics.quiescenceDepth ? configuredHeuristics
    : normalizeHeuristics({ ...configuredHeuristics, quiescenceDepth: qDepth });
  const timeMs = finiteOption(options.timeMs, 3000, 0, 3_600_000);
  const maxDepth = Math.floor(finiteOption(options.maxDepth, 8, 1, 64));
  const maxNodes = Math.floor(finiteOption(options.maxNodes, 2_000_000, 0, 1_000_000_000));
  const maxTableEntries = Math.floor(finiteOption(options.maxTableEntries, 100_000, 0, 1_000_000));
  const cacheMemoryMb = finiteOption(options.cacheMemoryMb, 128, 0, 4096);
  const deadline = options.unlimitedTime === true ? Infinity : started + timeMs;
  const tt = new SearchCache(maxTableEntries, Math.floor(cacheMemoryMb * 1024 * 1024));
  const history = new Map(), killers = new Map();
  const evalCache = new WeakMap(), checkCache = new WeakMap(), moveCache = new WeakMap();
  const policyPruned = new WeakSet();
  const keyPosition = createPositionKeyCache();
  let nodes = 0, searchNodes = 0, generationNodes = 0, qnodes = 0, ttHits = 0, qTtHits = 0, cutoffs = 0;
  let policyLeaves = 0;
  let interruption = null, depth = 0, bestAction = null, pv = [], score = null;
  let completed = false, status = 'incomplete', rootPartial = null;
  let activeQDepth = 0, completedQDepth = 0;
  let searchingDepth = 0, rootActionsSearched = 0, selectiveDepth = 0, lastProgress = started;
  const rootSign = colorSign(position);

  function tick(kind = 'generation') {
    if (options.shouldStop?.()) { interruption = 'cancelled'; throw new SearchInterrupted(); }
    if (nodes >= maxNodes) { interruption = 'nodes'; throw new SearchInterrupted(); }
    const now = performance.now();
    if (now >= deadline) { interruption = 'time'; throw new SearchInterrupted(); }
    if (options.claimNode && !options.claimNode(kind)) { interruption = 'nodes'; throw new SearchInterrupted(); }
    nodes++;
    if (kind === 'generation') generationNodes++;
    else { searchNodes++; if (kind === 'quiescence') qnodes++; }
    if (options.onProgress && bestAction !== null && now - lastProgress >= 250) {
      lastProgress = now;
      options.onProgress(snapshot());
    }
  }
  function staticScore(pos) {
    if (!evalCache.has(pos)) evalCache.set(pos, Math.max(-MATE_THRESHOLD + 1, Math.min(MATE_THRESHOLD - 1, evaluate(pos, heuristics) * colorSign(pos))));
    return evalCache.get(pos);
  }
  function checked(pos) {
    if (!checkCache.has(pos)) checkCache.set(pos, inCheck(pos));
    return checkCache.get(pos);
  }
  function orderingFeatures(pos, move) {
    // Generated move objects are shared across partial-turn siblings. Their
    // mover, target and coordinates cannot change within that turn; only the
    // learned ordering bonuses below need refreshing on each visit.
    if (!moveCache.has(move)) {
      const from = move[0], to = move[1];
      const centralGain = Math.abs(from[2] - 3.5) + Math.abs(from[3] - 3.5) - Math.abs(to[2] - 3.5) - Math.abs(to[3] - 3.5);
      moveCache.set(move, { ...moveFeatures(pos, move, heuristics), key: moveKey(move), history: historyKey(pos, move), centralGain });
    }
    return moveCache.get(move);
  }
  function orderMoves(pos, moves, preferred, ply, spatialFirst = false) {
    const favorites = new Set((preferred || []).map(moveKey));
    const killerMoves = new Set((killers.get(ply) || []).flat().map(moveKey));
    return moves.map((move, index) => {
      const f = orderingFeatures(pos, move);
      let priority = (favorites.has(f.key) ? 10_000_000 : 0) + f.promotion * 100;
      if (f.isCapture) priority += 1_000_000 + f.captureValue * 100 - pieceValue(f.mover, heuristics);
      else priority += (killerMoves.has(f.key) ? heuristics.killerBonus : 0) + (history.get(f.history) || 0) + f.centralGain * heuristics.quietCentralization;
      // Unforced early branching expands the reply tree enormously. Explore
      // ordinary development before speculative travel unless it wins material.
      if (f.temporal) priority -= spatialFirst ? heuristics.temporalMovePenalty : 100;
      return { move, priority, index };
    }).sort((a, b) => b.priority - a.priority || a.index - b.index).map(item => item.move);
  }
  function actions(pos, preferred, ply, { spatialFirst = true, tacticalOnly = false, restricted = true } = {}) {
    const iterator = generateActions(pos, {
      tick: () => tick(), tacticalOnly, preferredAction: preferred, keyPosition, skipOptionalSpatial: restricted,
      onSkipOptionalSpatial: () => policyPruned.add(iterator),
      orderMoves: (current, moves) => orderMoves(current, moves, preferred, ply, spatialFirst),
    });
    return iterator;
  }
  function rememberCutoff(pos, action, ply, remaining) {
    cutoffs++;
    if (action.some(move => { const f = moveFeatures(pos, move, heuristics); return f.isCapture || f.isPromotion; })) return;
    const list = killers.get(ply) || [];
    const key = actionKey(action);
    killers.set(ply, [action, ...list.filter(a => actionKey(a) !== key)].slice(0, 2));
    for (const move of action) {
      const key = historyKey(pos, move);
      history.set(key, Math.min(50_000, (history.get(key) || 0) + remaining * remaining * heuristics.historyBonus));
    }
  }
  function store(key, entry) {
    tt.store(key, entry);
  }
  function terminal(pos, ply) { return checked(pos) ? -MATE_SCORE + ply : 0; }
  function emptyResult(pos, ply, iterator) {
    // An exhausted traversal that excluded nothing already proves terminal.
    if (!policyPruned.has(iterator)) return { score: terminal(pos, ply), pv: [], terminal: true };
    // Policy exhaustion is not checkmate or stalemate. An unrestricted witness
    // is used only to validate terminal status, never as a searched candidate
    // or fallback action. Internally it establishes a static policy boundary.
    const witness = actions(pos, null, ply, { restricted: false });
    const next = witness.next();
    witness.return?.();
    if (next.done) return { score: terminal(pos, ply), pv: [], terminal: true };
    policyLeaves++;
    return { score: staticScore(pos), pv: [], policy: true };
  }

  function quiescence(pos, alpha, beta, remaining, ply) {
    selectiveDepth = Math.max(selectiveDepth, ply);
    tick('quiescence');
    // No continuation can lose before this ply or win before the next one.
    // These are mathematical bounds, including across multiboard turns: a
    // completed submission is one ply regardless of its component moves.
    alpha = Math.max(alpha, -MATE_SCORE + ply);
    beta = Math.min(beta, MATE_SCORE - ply - 1);
    if (alpha >= beta) return { score: alpha, pv: [] };
    // Keep each tactical horizon separate: a quiet warmup is not an exact
    // result for a later pass that searches recaptures. Share the bounded table
    // with normal search, but never reuse a tactical score as a full-turn one.
    const key = `q${remaining}:${keyPosition(pos)}`, entry = tt.get(key);
    if (entry) {
      ttHits++; qTtHits++;
      const value = fromTable(entry.score, ply);
      if (entry.flag === 'exact') return { score: value, pv: entry.pv };
      if (entry.flag === 'lower') alpha = Math.max(alpha, value);
      else beta = Math.min(beta, value);
      if (alpha >= beta) return { score: value, pv: entry.pv };
    }
    const searchAlpha = alpha, searchBeta = beta;
    function finish(value, pv = [], exact = false) {
      const flag = exact ? 'exact' : value <= searchAlpha ? 'upper' : value >= searchBeta ? 'lower' : 'exact';
      store(key, { depth: remaining, score: toTable(value, ply), flag, pv });
      return { score: value, pv };
    }
    const isCheck = remaining >= 0 && checked(pos);
    let best = isCheck || remaining < 0 ? -INF : staticScore(pos), bestPv = [];
    // A searched tactical action is also a witness that the position is not
    // terminal. Reuse it instead of constructing and abandoning a separate
    // legal turn first, which is costly when several boards must be played.
    const tacticalOnly = remaining > 0 && !isCheck && best < beta;
    let iterator = actions(pos, entry?.pv[0], ply, { tacticalOnly });
    let next = iterator.next();
    if (next.done && tacticalOnly) {
      iterator = actions(pos, null, ply);
      const witness = iterator.next();
      iterator.return?.();
      // No captures does not prove stalemate: quiet legal turns still count.
      if (witness.done) return finish(emptyResult(pos, ply, iterator).score, [], true);
      return finish(best);
    }
    // Prove at least one legal action before returning stand-pat: otherwise
    // stalemate or mate at the horizon could be mistaken for material gain.
    if (next.done) return finish(emptyResult(pos, ply, iterator).score, [], true);
    // One extra real evasion is allowed at a checked horizon. Its resulting
    // position still needs a terminal proof before static evaluation: an
    // evasion can itself deliver mate or stalemate. Do not extend check chains.
    if (remaining < 0) {
      iterator.return?.();
      return finish(staticScore(pos), [], true);
    }
    if (!isCheck) {
      if (best >= beta || remaining <= 0) { iterator.return?.(); return finish(best, [], remaining <= 0); }
      alpha = Math.max(alpha, best);
    }
    while (!next.done) {
      const candidate = next.value;
      selectiveDepth = Math.max(selectiveDepth, ply + 1);
      // A checked horizon searches actual legal evasions, never stand-pat.
      // Stop after that evasion at the configured boundary: extending four
      // further checked turns can explode the number of boards in 5D chess
      // and consume the entire budget before normal-depth search advances.
      const child = quiescence(candidate.position, -beta, -alpha, remaining - 1, ply + 1);
      const value = -child.score;
      if (value > best) { best = value; bestPv = [candidate.moves, ...child.pv]; }
      alpha = Math.max(alpha, value);
      if (alpha >= beta) { cutoffs++; iterator.return?.(); break; }
      next = iterator.next();
    }
    return finish(best, bestPv);
  }

  function negamax(pos, remaining, alpha, beta, ply, preferred = null) {
    selectiveDepth = Math.max(selectiveDepth, ply);
    if (remaining <= 0) return quiescence(pos, alpha, beta, activeQDepth, ply);
    if (ply === 0) rootActionsSearched = 0;
    tick('search');
    alpha = Math.max(alpha, -MATE_SCORE + ply);
    beta = Math.min(beta, MATE_SCORE - ply - 1);
    if (alpha >= beta) return { score: alpha, pv: [] };
    const key = keyPosition(pos), entry = tt.get(key);
    if (entry && entry.depth >= remaining && entry.quiescenceDepth >= activeQDepth && ply > 0) {
      ttHits++;
      const value = fromTable(entry.score, ply);
      if (entry.flag === 'exact') return { score: value, pv: entry.pv };
      if (entry.flag === 'lower') alpha = Math.max(alpha, value);
      else beta = Math.min(beta, value);
      if (alpha >= beta) return { score: value, pv: entry.pv };
    }
    // Classify the resulting bound against the window actually searched. A
    // cutoff against a TT-tightened beta must never be stored as exact.
    const searchAlpha = alpha, searchBeta = beta;
    let best = -INF, bestPv = [], bestMove = null, count = 0;
    const iterator = actions(pos, preferred || entry?.bestAction, ply);
    for (const candidate of iterator) {
      let child;
      if (!count) child = negamax(candidate.position, remaining - 1, -beta, -alpha, ply + 1);
      else {
        child = negamax(candidate.position, remaining - 1, -alpha - 1, -alpha, ply + 1);
        const probe = -child.score;
        if (probe > alpha && probe < beta) child = negamax(candidate.position, remaining - 1, -beta, -alpha, ply + 1);
      }
      count++;
      if (ply === 0) rootActionsSearched = count;
      const value = -child.score;
      if (value > best) {
        best = value; bestMove = candidate.moves; bestPv = [candidate.moves, ...child.pv];
        if (ply === 0) rootPartial = { score: best, bestAction: bestMove, pv: bestPv };
      }
      alpha = Math.max(alpha, value);
      if (alpha >= beta) { rememberCutoff(pos, candidate.moves, ply, remaining); break; }
    }
    if (!count) return emptyResult(pos, ply, iterator);
    const flag = best <= searchAlpha ? 'upper' : best >= searchBeta ? 'lower' : 'exact';
    store(key, { depth: remaining, quiescenceDepth: activeQDepth, score: toTable(best, ply), flag, bestAction: bestMove, pv: bestPv });
    return { score: best, pv: bestPv };
  }

  function snapshot() {
    const elapsedMs = Math.max(0, performance.now() - started);
    return {
      bestAction, score: score === null ? null : Math.round(score * rootSign), depth,
      nodes, searchNodes, generationNodes, qnodes, ttHits, qTtHits, cutoffs, elapsedMs: Math.round(elapsedMs),
      searchingDepth, rootActionsSearched, selectiveDepth,
      nps: elapsedMs ? Math.round(nodes * 1000 / elapsedMs) : 0, pv, status, completed,
      stoppedReason: interruption, tableEntries: tt.size, cacheMemoryBytes: tt.memoryBytes,
      searchPolicy: 'present-spatial', policyLeaves, heuristics,
      effectiveQuiescenceDepth: completed ? completedQDepth : activeQDepth,
      scoreType: score === null ? 'unavailable' : Math.abs(score) > MATE_THRESHOLD ? 'mate' : 'cp',
      mateIn: score !== null && Math.abs(score) > MATE_THRESHOLD ? Math.sign(score * rootSign) * (MATE_SCORE - Math.abs(score)) : null,
      limits: { timeMs, maxDepth, maxNodes, quiescenceDepth: qDepth, cacheMemoryMb, maxTableEntries }
    };
  }
  function* iterations() {
    try {
      const fallbackIterator = actions(position, null, 0);
      const fallback = fallbackIterator.next();
      fallbackIterator.return?.();
      if (fallback.done) {
        const result = emptyResult(position, 0, fallbackIterator);
        if (result.policy) interruption = 'policy';
        else { score = result.score; status = score ? 'checkmate' : 'stalemate'; completed = true; }
        return snapshot();
      }
      bestAction = fallback.value.moves; pv = [bestAction];
      // A legal fallback is valuable even if the first recursive iteration cannot
      // finish; its score stays explicitly unavailable until a child is searched.
      const iterations = [];
      // First broaden capture analysis at depth one; then deepen full turns. A
      // quiet warmup alone can overvalue a defended capture, so finish q1 before
      // investing in a much larger depth-two multiverse tree.
      if (options.quiescenceWarmup !== false) {
        for (let horizon = 0; horizon < qDepth; horizon++) iterations.push({ depth: 1, horizon });
      }
      for (let currentDepth = 1; currentDepth <= maxDepth; currentDepth++) iterations.push({ depth: currentDepth, horizon: qDepth });
      for (const iteration of iterations) {
        const currentDepth = iteration.depth;
        searchingDepth = currentDepth;
        activeQDepth = iteration.horizon;
        rootPartial = null;
        const window = currentDepth > 1 && heuristics.aspirationWindow > 0 && Math.abs(score ?? 0) < MATE_THRESHOLD ? heuristics.aspirationWindow : INF;
        const lower = window === INF ? -INF : score - window;
        const upper = window === INF ? INF : score + window;
        let result = yield { remaining: currentDepth, horizon: activeQDepth, alpha: lower, beta: upper, preferred: bestAction };
        if (result.score <= lower || result.score >= upper) result = yield { remaining: currentDepth, horizon: activeQDepth, alpha: -INF, beta: INF, preferred: result.pv[0] || bestAction };
        score = result.score; depth = currentDepth; pv = result.pv; bestAction = pv[0] || bestAction;
        completedQDepth = activeQDepth;
        completed = true; status = 'ok';
        lastProgress = performance.now();
        options.onProgress?.(snapshot());
        if (Math.abs(score) >= MATE_SCORE - currentDepth) { interruption = 'mate'; break; }
      }
      if (!interruption) interruption = 'depth';
    } catch (error) {
      if (!(error instanceof SearchInterrupted)) throw error;
      if (!completed && rootPartial) {
        bestAction = rootPartial.bestAction; score = rootPartial.score; pv = rootPartial.pv;
      }
    }
    return snapshot();
  }

  // Both drivers use the same recursive search, ordering and bound semantics.
  // A worker keeps this context alive across jobs and iterative depths.
  return {
    iterations, snapshot,
    root: request => negamax(position, request.remaining, request.alpha, request.beta, 0, request.preferred),
    subtree(request) {
      activeQDepth = request.horizon;
      return negamax(request.position, request.remaining, request.alpha, request.beta, request.ply, request.preferred);
    },
    beginRoot(request) {
      rootActionsSearched = 0;
      rootPartial = null;
      tick('search');
      return actions(position, request.preferred, 0);
    },
    acceptRoot(candidate, child, count) {
      rootActionsSearched = count;
      const value = -child.score;
      if (!rootPartial || value > rootPartial.score) {
        rootPartial = { score: value, bestAction: candidate.moves, pv: [candidate.moves, ...child.pv] };
      }
    },
    emptyRoot: iterator => emptyResult(position, 0, iterator),
    interrupt(reason) { interruption = reason; return new SearchInterrupted(); },
    isInterrupted: error => error instanceof SearchInterrupted,
  };
}

export function analyze(position, options = {}) {
  const session = createSearchSession(position, options), iterations = session.iterations();
  let step = iterations.next();
  while (!step.done) {
    let result;
    try { result = session.root(step.value); }
    catch (error) { return iterations.throw(error).value; }
    step = iterations.next(result);
  }
  return step.value;
}
