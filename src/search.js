import { applyMove, presentTimelines, createPositionKeyCache, createSearchMoveGenerator, createSearchRoyalSafety, generateActions } from './rules.js';
import { createEvaluator, pieceValuesFor } from './evaluate.js';
import { normalizeHeuristics } from './heuristics.js';
import { SearchCache } from './search-cache.js';

export const MATE_SCORE = 100_000;
const MATE_THRESHOLD = MATE_SCORE - 1000;
const INF = 1_000_000;
const NO_MOVES = Object.freeze([]);
class SearchInterrupted extends Error {}
const actionKey = action => JSON.stringify(action);
const moveKey = move => JSON.stringify(move);
const colorSign = position => position.action % 2 === 0 ? 1 : -1;
const toTable = (score, ply) => score > MATE_THRESHOLD ? score + ply : score < -MATE_THRESHOLD ? score - ply : score;
const fromTable = (score, ply) => score > MATE_THRESHOLD ? score - ply : score < -MATE_THRESHOLD ? score + ply : score;

function orderingLookup(moves) {
  // A preferred turn and two killers usually contain only a few components.
  // Keep constant-time lookup for unusually large multiverse turns.
  return moves.length > 8 ? new Set(moves.map(moveKey)) : moves;
}

function hasOrderingMove(lookup, move) {
  if (lookup instanceof Set) return lookup.has(moveKey(move));
  candidate: for (let index = 0; index < lookup.length; index++) {
    const other = lookup[index];
    if (move === other) return true;
    if (move.length !== other.length) continue;
    for (let component = 0; component < move.length; component++) {
      const left = move[component], right = other[component];
      if (left.length !== right.length) continue candidate;
      for (let axis = 0; axis < left.length; axis++) if (left[axis] !== right[axis]) continue candidate;
    }
    return true;
  }
  return false;
}

function finiteOption(value, fallback, min, max) {
  return Number.isFinite(Number(value)) ? Math.max(min, Math.min(max, Number(value))) : fallback;
}

function tacticalMove(position, move) {
  const to = move[1];
  return to.length > 4 || move.length === 3 || !!position.board[to[0]]?.[to[1]]?.[to[2]]?.[to[3]];
}

function historyKey(position, move) {
  const [from, to] = move;
  const piece = Math.abs(position.board[from[0]]?.[from[1]]?.[from[2]]?.[from[3]] || 0);
  // Ordinary quiet moves fit in five base-16 fields on every supported board
  // (up to 16 x 16). Numeric keys avoid allocating the same history string at
  // every ply. Temporal moves and promotions retain their full string keys.
  if (from[0] === to[0] && from[1] === to[1] && !to[4]
      && from[2] < 16 && from[3] < 16 && to[2] < 16 && to[3] < 16) {
    return ((((piece * 16 + from[2]) * 16 + from[3]) * 16 + to[2]) * 16 + to[3]);
  }
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
  const pieceValues = pieceValuesFor(heuristics);
  const evaluate = createEvaluator(heuristics), royalSafety = createSearchRoyalSafety();
  const generateMoves = createSearchMoveGenerator();
  const timeMs = finiteOption(options.timeMs, 3000, 0, 3_600_000);
  const maxDepth = Math.floor(finiteOption(options.maxDepth, 8, 1, 64));
  const maxNodes = Math.floor(finiteOption(options.maxNodes, 2_000_000, 0, 1_000_000_000));
  const maxTableEntries = Math.floor(finiteOption(options.maxTableEntries, 100_000, 0, 1_000_000));
  const cacheMemoryMb = finiteOption(options.cacheMemoryMb, 128, 0, 4096);
  const deadline = options.unlimitedTime === true ? Infinity : started + timeMs;
  const tt = new SearchCache(maxTableEntries, Math.floor(cacheMemoryMb * 1024 * 1024));
  const history = new Map(), killers = new Map();
  const evalCache = new WeakMap(), checkCache = new WeakMap();
  const policyPruned = new WeakSet();
  const keyPosition = createPositionKeyCache({ compact: true });
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
    let value = evalCache.get(pos);
    if (value === undefined) {
      value = Math.max(-MATE_THRESHOLD + 1, Math.min(MATE_THRESHOLD - 1, evaluate(pos) * colorSign(pos)));
      evalCache.set(pos, value);
    }
    return value;
  }
  function checked(pos) {
    let value = checkCache.get(pos);
    if (value === undefined) {
      value = royalSafety.inCheck(pos);
      checkCache.set(pos, value);
    }
    return value;
  }
  function* orderMoves(pos, moves, favorites, killerMoves, spatialFirst = false) {
    if (!moves.length) return;
    // Keep only the numeric priorities for this visit. Retaining a feature
    // object and weak-cache entry for every generated move costs more than
    // recomputing these few scalars, especially when most moves get cut off.
    const priorities = new Float64Array(moves.length);
    let first = 0;
    for (let index = 0; index < moves.length; index++) {
      const move = moves[index];
      const [from, to] = move;
      const captured = pos.board[to[0]]?.[to[1]]?.[to[2]]?.[to[3]] || 0;
      // A third coordinate denotes en passant; castling has four coordinates.
      const isCapture = !!captured || move.length === 3, isPromotion = to.length > 4;
      const mover = isCapture || isPromotion ? pos.board[from[0]]?.[from[1]]?.[from[2]]?.[from[3]] || 0 : 0;
      const moverValue = pieceValues[Math.ceil(Math.abs(mover) / 2)] || 0;
      const promotion = isPromotion ? (pieceValues[Math.ceil(Math.abs(to[4]) / 2)] || 0) - moverValue : 0;
      // Compare short preferred/killer lists directly, including all auxiliary
      // coordinates. Most moves need no serialized key for either lookup.
      let priority = (hasOrderingMove(favorites, move) ? 10_000_000 : 0) + promotion * 100;
      if (isCapture) {
        const captureValue = pieceValues[Math.ceil(Math.abs(captured) / 2)] || (move.length === 3 ? pieceValues[1] : 0);
        priority += 1_000_000 + captureValue * 100 - moverValue;
      } else priority += (hasOrderingMove(killerMoves, move) ? heuristics.killerBonus : 0)
        + (history.size ? history.get(historyKey(pos, move)) || 0 : 0)
        + (Math.abs(from[2] - 3.5) + Math.abs(from[3] - 3.5) - Math.abs(to[2] - 3.5) - Math.abs(to[3] - 3.5)) * heuristics.quietCentralization;
      // Unforced early branching expands the reply tree enormously. Explore
      // ordinary development before speculative travel unless it wins material.
      if (from[0] !== to[0] || from[1] !== to[1]) priority -= spatialFirst ? heuristics.temporalMovePenalty : 100;
      priorities[index] = priority;
      if (priority > priorities[first]) first = index;
    }
    yield moves[first];
    // A beta cutoff often needs only the best component. Defer sorting and
    // allocating its remaining indices until another component is requested.
    // Numeric snapshots remain stable when deeper sibling visits update
    // history while this iterator is suspended.
    const ordered = [];
    for (let index = 0; index < moves.length; index++) if (index !== first) ordered.push(index);
    ordered.sort((a, b) => priorities[b] - priorities[a]);
    for (const index of ordered) yield moves[index];
  }
  function actions(pos, preferred, ply, { spatialFirst = true, tacticalOnly = false, restricted = true, firstOnly = false, ordered = true } = {}) {
    // A checked multiverse can have a one-move escape even when several boards
    // are required: a temporal arrival can advance two boards, or a branch can
    // move the present into the past. Find that reply before the depth-first
    // generator combines unrelated components on every other board. This only
    // supplies an ordering hint; normal generation still searches all turns.
    if (!preferred && !tacticalOnly && pos.board.length > 1 && checked(pos)
        && presentTimelines(pos).length > 1) {
      tick();
      for (const move of generateMoves(pos)) {
        // A spatial move cannot complete several required boards. Temporal
        // moves are permitted from optional boards under either search policy.
        if (move[0][0] === move[1][0] && move[0][1] === move[1][1]) continue;
        tick();
        const next = applyMove(pos, move);
        if (presentTimelines(next).length === 0 && !royalSafety.attackedByNextPlayer(next)) {
          preferred = [move];
          break;
        }
      }
    }
    // The iterator is suspended while deeper plies search; only those deeper
    // plies can update their killers. Snapshot these lists once for a whole turn,
    // including its many alternative component sequences.
    let favorites, killerMoves;
    const iterator = generateActions(pos, {
      tick: () => tick(), tacticalOnly, firstOnly, preferredAction: preferred, keyPosition, generateMoves, royalSafety, skipOptionalSpatial: restricted,
      onSkipOptionalSpatial: () => policyPruned.add(iterator),
      orderMoves: ordered ? (current, moves) => orderMoves(current, moves,
        favorites ??= orderingLookup(preferred || NO_MOVES),
        killerMoves ??= orderingLookup(killers.get(ply)?.flat() || NO_MOVES), spatialFirst) : undefined,
    });
    return iterator;
  }
  function rememberCutoff(pos, action, ply, remaining) {
    cutoffs++;
    if (action.some(move => tacticalMove(pos, move))) return;
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
    const witness = actions(pos, null, ply, { restricted: false, firstOnly: true });
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
    const positionKey = keyPosition(pos), key = `q${remaining}:${positionKey}`, entry = tt.get(key);
    if (entry) {
      ttHits++; qTtHits++;
      const value = fromTable(entry.score, ply);
      if (entry.flag === 'exact') return { score: value, pv: entry.pv };
      if (entry.flag === 'lower') alpha = Math.max(alpha, value);
      else beta = Math.min(beta, value);
      if (alpha >= beta) return { score: value, pv: entry.pv };
    }
    // Legal-turn existence is independent of the tactical horizon and search
    // window. Reuse that proof from warmup or an earlier tactical visit, while
    // keeping their scores separate. In particular, a policy-exhausted leaf
    // must never be recorded as a legal witness.
    const warmup = !entry && remaining !== 0 ? tt.get(`q0:${positionKey}`) : null;
    let hasLegalAction = !!(entry?.hasLegalAction || warmup?.hasLegalAction);
    const cachedCheck = entry?.inCheck ?? warmup?.inCheck;
    const cachedStatic = entry?.staticScore ?? warmup?.staticScore;
    if (cachedCheck !== undefined) checkCache.set(pos, cachedCheck);
    if (cachedStatic !== undefined) evalCache.set(pos, cachedStatic);
    const searchAlpha = alpha, searchBeta = beta;
    function finish(value, pv = [], exact = false) {
      const flag = exact ? 'exact' : value <= searchAlpha ? 'upper' : value >= searchBeta ? 'lower' : 'exact';
      store(key, { depth: remaining, score: toTable(value, ply), flag, pv, hasLegalAction,
        inCheck: checkCache.get(pos), staticScore: evalCache.get(pos) });
      return { score: value, pv };
    }
    const isCheck = remaining >= 0 && checked(pos);
    let best = isCheck || remaining < 0 ? -INF : staticScore(pos), bestPv = [];
    if (hasLegalAction) {
      if (remaining < 0) return finish(staticScore(pos), [], true);
      if (!isCheck && (best >= beta || remaining === 0)) return finish(best, [], remaining === 0);
    }
    // A searched tactical action is also a witness that the position is not
    // terminal. Reuse it instead of constructing and abandoning a separate
    // legal turn first, which is costly when several boards must be played.
    const tacticalOnly = remaining > 0 && !isCheck && best < beta;
    const firstOnly = remaining < 0 || (!isCheck && (best >= beta || remaining === 0));
    // An unchecked single-timeline static boundary only needs one legal move.
    // Its witness never enters the PV or updates ordering history. Avoid full
    // ranking here; keep it for multiboard combinations and checked boundaries,
    // where a poor first component can make the existence probe expensive.
    const ordered = !firstOnly || pos.board.length !== 1 || remaining < 0;
    let iterator = actions(pos, entry?.pv[0], ply, { tacticalOnly, firstOnly, ordered });
    let next = iterator.next();
    if (next.done && tacticalOnly) {
      if (hasLegalAction) return finish(best, [], true);
      iterator = actions(pos, null, ply, { firstOnly: true, ordered: pos.board.length !== 1 });
      const witness = iterator.next();
      iterator.return?.();
      // No captures does not prove stalemate: quiet legal turns still count.
      if (witness.done) return finish(emptyResult(pos, ply, iterator).score, [], true);
      hasLegalAction = true;
      return finish(best, [], true);
    }
    // Prove at least one legal action before returning stand-pat: otherwise
    // stalemate or mate at the horizon could be mistaken for material gain.
    if (next.done) return finish(emptyResult(pos, ply, iterator).score, [], true);
    hasLegalAction = true;
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
    // A failed aspiration pass also leaves a valid root bound. Reuse it when
    // retrying that depth, just as for an internal principal-variation probe.
    if (entry && entry.depth >= remaining && entry.quiescenceDepth >= activeQDepth) {
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
    if (!count) {
      const result = emptyResult(pos, ply, iterator);
      store(key, { depth: remaining, quiescenceDepth: activeQDepth,
        score: toTable(result.score, ply), flag: 'exact', pv: [] });
      return result;
    }
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
      const fallbackIterator = actions(position, null, 0, { firstOnly: true });
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
    // Root workers report counters after each candidate. Omit result metadata,
    // configuration, and PV arrays from these frequent cross-thread messages.
    statistics: () => ({ searchNodes, generationNodes, qnodes, ttHits, qTtHits, cutoffs,
      policyLeaves, tableEntries: tt.size, cacheMemoryBytes: tt.memoryBytes, selectiveDepth }),
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
