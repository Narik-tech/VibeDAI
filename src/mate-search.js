import { applyMove, presentTimelines, createPositionKeyCache, createSearchMoveGenerator, createSearchRoyalSafety, generateActions } from './rules.js';

/**
 * Prove a mate in one complete turn, including optional component moves.
 * A local limit means unknown; only unrestricted exhaustion rules out mate.
 * The caller's tick can enforce a shared budget/cancellation and its errors
 * propagate unchanged. Keep only the winning action, never candidate boards.
 * As in generateActions, orderMoves must retain every supplied component.
 */
export function scanMateInOne(position, {
  maxWork = 2048, timeMs = 25, tick = () => {},
  keyPosition = createPositionKeyCache({ compact: true }),
  generateMoves = createSearchMoveGenerator(), royalSafety = createSearchRoyalSafety(),
  orderMoves, preferredAction = null, preferPresent = true,
} = {}) {
  if (!(maxWork === Infinity || Number.isSafeInteger(maxWork) && maxWork >= 0)) throw new Error('Mate probe work must be a nonnegative integer.');
  if (!(typeof timeMs === 'number' && timeMs >= 0)) throw new Error('Mate probe time must be a nonnegative number.');
  const deadline = performance.now() + timeMs;
  const localLimit = new Error('Mate probe budget exhausted.');
  let work = 0, candidates = 0, checks = 0, reason = null;
  const result = (status, action = null) => ({ status, action, work, candidates, checks, reason });
  const charge = () => {
    if (work >= maxWork) { reason = 'work'; throw localLimit; }
    if (performance.now() >= deadline) { reason = 'time'; throw localLimit; }
    tick();
    work++;
  };
  const options = {
    tick: charge, keyPosition, generateMoves, royalSafety, orderMoves,
    skipOptionalSpatial: false, tacticalOnly: false,
  };
  const hasQuickReply = current => {
    // A temporal move can advance several required boards at once. Find that
    // witness before depth-first reply generation combines unrelated moves.
    if (current.board.length === 1 || presentTimelines(current).length < 2) return false;
    charge();
    for (const move of generateMoves(current)) {
      if (move[0][0] === move[1][0] && move[0][1] === move[1][1]) continue;
      charge();
      const next = applyMove(current, move);
      if (presentTimelines(next).length === 0 && !royalSafety.attackedByNextPlayer(next)) return true;
    }
    return false;
  };
  try {
    // Try ordinary present-board turns first, before optional continuations
    // multiply the attack tree. This is an ordering stage, never a negative
    // proof: every skipped turn remains eligible in the full-rule second pass.
    let skipped = false;
    const seen = new Set();
    for (const restricted of preferPresent ? [true, false] : [false]) {
      for (const candidate of generateActions(position, {
        ...options, preferredAction, skipOptionalSpatial: restricted,
        onSkipOptionalSpatial: () => { skipped = true; },
      })) {
        const key = keyPosition(candidate.position);
        if (seen.has(key)) continue;
        seen.add(key);
        candidates++;
        charge();
        if (!royalSafety.inCheck(candidate.position)) continue;
        checks++;
        if (hasQuickReply(candidate.position)) continue;
        const replies = generateActions(candidate.position, { ...options, firstOnly: true });
        try {
          if (replies.next().done) return result('mate', candidate.moves);
        } finally { replies.return(); }
      }
      if (!skipped) break;
    }
    return result('no-mate');
  } catch (error) {
    if (error !== localLimit) throw error;
    return result('unknown');
  }
}
