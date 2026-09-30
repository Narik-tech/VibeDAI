import { generateActions } from '../src/rules.js';

export const emptyCoverage = () => ({ actions: 0, temporal: 0, compound: 0, branching: 0 });

// Count actual legal submitted turns and the histories they produce. A temporal
// move need not create a new timeline, so report branching separately.
export function actionCoverage(position, action, next) {
  return { actions: 1,
    temporal: Number(action.some(([from, to]) => from[0] !== to[0] || from[1] !== to[1])),
    compound: Number(action.length > 1),
    branching: Number(next.board.filter(Boolean).length > position.board.filter(Boolean).length) };
}

export function addCoverage(total, coverage) {
  for (const key of Object.keys(total)) total[key] += coverage[key] ?? 0;
  return total;
}

/** Bounded sampling from a seeded randomized traversal, rather than a fixed
 * prefix of the rules generator. This is not uniform over all legal turns. */
export function sampleExploration(position, random, { maxNodes, timeMs, check = () => {}, maxCandidates = 32 } = {}) {
  const deadline = performance.now() + timeMs, exhausted = new Error('Exploration budget exhausted.');
  const candidates = [], coverage = emptyCoverage();
  let work = 0, stoppedReason = null, exhaustive = false;
  const tick = () => {
    check();
    if (work >= maxNodes) { stoppedReason = 'exploration-work-limit'; throw exhausted; }
    if (performance.now() >= deadline) { stoppedReason = 'exploration-time-limit'; throw exhausted; }
    work++;
  };
  const legal = generateActions(position, { pruneUnsafe: false, cacheMoves: false, skipOptionalSpatial: false,
    orderMoves(_position, moves) {
      const shuffled = moves.slice();
      for (let index = shuffled.length - 1; index > 0; index--) {
        tick();
        const other = Math.floor(random() * (index + 1));
        [shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]];
      }
      return shuffled;
    },
    tick,
  });
  try {
    while (candidates.length < maxCandidates) {
      const next = legal.next();
      if (next.done) { exhaustive = true; break; }
      candidates.push(next.value);
      addCoverage(coverage, actionCoverage(position, next.value.moves, next.value.position));
    }
    if (!exhaustive) stoppedReason = 'candidate-limit';
  } catch (error) { if (error !== exhausted) throw error; }
  finally { legal.return?.(); }
  check();
  const selected = candidates.length ? candidates[Math.floor(random() * candidates.length)] : null;
  return { action: selected?.moves ?? null, candidates: candidates.length, work, exhaustive, stoppedReason,
    sampling: 'seeded-component-shuffle', coverage,
    selectedCoverage: selected ? actionCoverage(position, selected.moves, selected.position) : emptyCoverage() };
}
