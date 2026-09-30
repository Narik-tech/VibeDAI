import { applyMove, pseudoMoves, raw, validateAction } from './rules.js';

export const COMPONENT_POLICY_VERSION = 1;
export const MAX_POLICY_MOVES = 16384;
export const SOFT_POLICY_TARGET_VERSION = 1;

/** Teacher targets use the same required-or-temporal component mask as search.
 * Every target is a component of a validated complete teacher turn. Pseudo-legal
 * alternatives remain in the mask: legality of a full compound turn still
 * belongs to rules/search, not the policy head.
 */
export function componentPolicyTargets(position, action) {
  if (!Array.isArray(action) || action.length === 0 || action.length > 256) return [];
  validateAction(position, action);
  const examples = [];
  let current = position;
  for (const [componentIndex, selected] of action.entries()) {
    const present = raw.boardFuncs.present(current.board, current.action);
    const moves = pseudoMoves(current).filter(([from, to]) => present.includes(from[0])
      || from[0] !== to[0] || from[1] !== to[1]);
    const target = moves.findIndex(move => raw.validateFuncs.compareMove(move, selected) === 0);
    if (target >= 0 && moves.length <= MAX_POLICY_MOVES) {
      examples.push({ ...(current === position ? {} : { position: current }), moves, target, componentIndex });
    }
    current = applyMove(current, selected);
  }
  return examples;
}

/** Distill only searched complete turns. Zero weights mean unobserved, not bad.
 * Each component receives its best evaluated continuation, avoiding a preference
 * for components merely searched in more move orders. Legacy teachers without
 * policyAlternatives retain the original selected-action labels.
 */
export function policyTargetsFromSearch(position, result) {
  const action = result?.bestAction;
  const examples = componentPolicyTargets(position, action);
  if (!Array.isArray(result?.policyAlternatives)) return examples;
  const alternatives = result.policyAlternatives.filter(item => Array.isArray(item?.action)
    && item.action.length > 0 && Number.isFinite(item.score));
  for (const alternative of alternatives) validateAction(position, alternative.action);
  const direction = position.action % 2 === 0 ? 1 : -1;
  const same = (a, b) => raw.validateFuncs.compareMove(a, b) === 0;
  return examples.flatMap(example => {
    const index = example.componentIndex;
    const scores = Array(example.moves.length).fill(null);
    for (const alternative of alternatives) {
      if (alternative.action.length <= index
          || action.slice(0, index).some((move, previous) => !same(move, alternative.action[previous]))) continue;
      const candidate = example.moves.findIndex(move => same(move, alternative.action[index]));
      if (candidate < 0) continue;
      const score = direction * alternative.score;
      scores[candidate] = scores[candidate] === null ? score : Math.max(scores[candidate], score);
    }
    const observed = scores.filter(score => score !== null);
    if (!observed.length) return [];
    const best = Math.max(...observed);
    const weights = scores.map(score => score === null ? 0 : Math.exp(Math.max(-20, (score - best) / 250)));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    return [{ ...example, targetVersion: SOFT_POLICY_TARGET_VERSION,
      targetWeights: weights.map(weight => weight / total) }];
  });
}
