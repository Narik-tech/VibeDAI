// One schema drives validation, the analysis controls, and search configuration.
// All scores are centipawns unless marked as a multiplier or ordering points.
const setting = (key, label, description, group, value, min, max, step, unit) =>
  Object.freeze({ key, label, description, group, default: value, min, max, step, unit });
const multiplier = (key, label, description, group) => setting(key, label, description, group, 1, 0, 4, 0.05, '×');
const piece = (key, label, value) => setting(key, label,
  `Material value of one ${label.toLowerCase()}. Also affects capture ordering and historical travel discounts.${['bishopValue', 'knightValue', 'rookValue', 'queenValue', 'princessValue'].includes(key) ? ' Counts toward middlegame material phase.' : ''}`,
  'Piece values', value, 0, 5000, 5, 'cp');

export const HEURISTIC_SETTINGS = Object.freeze([
  ...[
    ['material', 'Material', 'Weighted average of material on timeline frontiers. Historical copies never add material.'],
    ['activity', 'Activity', 'Weighted average of development, mobility, centralization, and pawn structure.'],
    ['kingSafety', 'King safety', 'Frontier shelter, persistent historical corridors, and the weakest active royal.'],
    ['temporal', 'Temporal pressure', 'Potential attacks on enemy royals across boards and time, capped before this multiplier.'],
    ['timelines', 'Timeline resources', 'Branch reserves, overextension, and material weaknesses across active boards.'],
    ['travel', 'Historical travel', 'The best historical capture route to an undefended pawn beside an enemy royal.'],
  ].map(([key, label, description]) => multiplier(`${key}Weight`, label, description, 'Components')),
  piece('pawnValue', 'Pawn', 100), piece('bishopValue', 'Bishop', 355), piece('knightValue', 'Knight', 340),
  piece('rookValue', 'Rook', 550), piece('queenValue', 'Queen', 1150), piece('princessValue', 'Princess', 900),
  piece('brawnValue', 'Brawn', 140), piece('commonKingValue', 'Common king', 370),
  piece('unicornValue', 'Unicorn', 450), piece('dragonValue', 'Dragon', 350),
  setting('phaseDivisor', 'Middlegame material scale', 'Middlegame factor is min(1, combined bishop, knight, rook, queen, and princess material divided by this value). Scales development, shelter, corridors, and royal centralization.', 'Position weighting', 6000, 100, 30000, 100, 'cp'),
  setting('inactiveWeight', 'Inactive timeline weight', 'Active frontier boards have weight 1. This is the weight of inactive frontiers in averaged evaluation and temporal opportunities.', 'Position weighting', 0.2, 0, 1, 0.05, '×'),
  setting('historicalPressureWeight', 'Historical royal weight', 'Historical royal targets receive this fraction of their timeline weight when estimating direct temporal pressure.', 'Position weighting', 0.65, 0, 1, 0.05, '×'),
  multiplier('pawnAdvanceWeight', 'Pawn advancement', 'Pawn and brawn advancement: 7r + 7 × max(0, r − 3)², with r measured from the home rank.', 'Pawn structure'),
  multiplier('pawnCenterWeight', 'Pawn centralization', 'Pawn and brawn centrality contributes 3 cp per unit; centrality follows actual board dimensions.', 'Pawn structure'),
  multiplier('doubledPawnWeight', 'Doubled pawns', 'Each pawn or brawn sharing its file with another friendly pawn or brawn loses 9 cp.', 'Pawn structure'),
  multiplier('isolatedPawnWeight', 'Isolated pawns', 'Each pawn or brawn without a friendly pawn or brawn on an adjacent file loses 9 cp.', 'Pawn structure'),
  multiplier('passedPawnWeight', 'Passed pawns', 'A pawn or brawn with no enemy pawn or brawn ahead on the same or adjacent file earns 8 + 2r² cp.', 'Pawn structure'),
  multiplier('knightCenterWeight', 'Knight centralization', 'Knights earn 11 cp per unit of board-aware centrality.', 'Activity'),
  multiplier('bishopCenterWeight', 'Bishop centralization', 'Bishops earn 6 cp per unit of board-aware centrality.', 'Activity'),
  multiplier('pieceCenterWeight', 'Other piece centralization', 'Other nonroyal, non-pawn pieces earn 3 cp per unit of board-aware centrality.', 'Activity'),
  multiplier('royalCenterWeight', 'Endgame royal centralization', 'Royals earn 10 × (1 − middlegame factor) cp per centrality unit.', 'Activity'),
  multiplier('mobilityWeight', 'Spatial mobility', 'Queens earn 2 cp per reachable square; other nonroyals earn 4. Uses ordinary board-plane reach without testing destination safety.', 'Activity'),
  multiplier('developmentWeight', 'Minor-piece development', 'Unmoved bishops and knights lose 12 × middlegame factor cp.', 'Activity'),
  multiplier('rookFileWeight', 'Rook files', 'Rooks earn 14 cp on files without a friendly pawn or brawn; enemy pawns do not remove the bonus.', 'Activity'),
  multiplier('shelterWeight', 'Royal pawn shield', 'Each missing forward shield pawn or brawn contributes 12 × middlegame factor cp of royal danger.', 'King safety'),
  multiplier('nearbyEnemyWeight', 'Nearby royal threats', 'Each enemy non-pawn within three squares contributes 8 cp of royal danger. Brawns count as threats.', 'King safety'),
  multiplier('flankSafetyWeight', 'Home-rank flank shelter', 'A royal near a home-rank flank reduces local danger by 15 cp. Local danger cannot fall below zero.', 'King safety'),
  multiplier('corridorWeight', 'Historical king-zone corridors', 'Time/rank and time/rank/file approaches add up to 24 cp per route, scaled by target importance and phase. Uses the worst sampled historical exposure.', 'King safety'),
  multiplier('worstKingWeight', 'Weakest active royal', 'Adds 45% of each side’s worst active royal danger, including local shelter and historical corridors, outside the frontier average.', 'King safety'),
  multiplier('temporalPressureWeight', 'Direct temporal attacks', 'Each frontier attacker contributes its best cross-board royal attack, up to 20 cp times board weight. The White-minus-Black pressure is capped at ±100 cp before component weighting.', 'Multiverse'),
  multiplier('reserveWeight', 'Branch reserves', 'Each unused active branch reserve contributes 90 cp, up to four per side. The first unmatched branch changes the reserve advantage by 180 cp.', 'Multiverse'),
  multiplier('overextensionWeight', 'Branch overextension', 'Branches beyond the opponent’s extent plus one cost 45 cp each, capped at three. Sparse and inactive timelines still consume capacity.', 'Multiverse'),
  multiplier('travelOpportunityWeight', 'Historical pawn entry', 'Best historical route per side to a king-adjacent pawn with no nonroyal spatial defender: up to 140 cp, discounted for low-value attackers and setups awaiting a reply. Requires branch reserve.', 'Multiverse'),
  multiplier('weakBoardWeight', 'Material weakness across boards', 'With multiple active boards, adds 12% of the lowest negative material balance plus the highest positive balance to preserve extreme imbalances.', 'Multiverse'),
  setting('quiescenceDepth', 'Tactical extension depth', 'Maximum extra complete turns containing a capture or promotion. Checked horizons also search an evasion turn.', 'Search', 2, 0, 8, 1, 'turns'),
  setting('aspirationWindow', 'Aspiration window', 'From depth two, initially search this far above and below the previous score; retry failures with a full window. Zero disables aspiration windows.', 'Search', 60, 0, 500, 5, 'cp'),
  setting('temporalMovePenalty', 'Temporal move ordering penalty', 'Subtract these ordering points from cross-board moves when spatial moves are preferred. Changes ordering, not evaluation or legality.', 'Search', 2000000, 0, 5000000, 10000, 'points'),
  setting('quietCentralization', 'Quiet centralization ordering', 'Ordering points per improvement in distance to the fixed 8×8 center (3.5, 3.5). Does not change evaluation.', 'Search', 10, 0, 100, 1, 'points'),
  setting('killerBonus', 'Quiet killer priority', 'Ordering bonus for quiet turns that previously caused a cutoff; the search remembers two per ply.', 'Search', 100000, 0, 500000, 1000, 'points'),
  setting('historyBonus', 'Quiet history learning', 'Quiet cutoff history increases by this coefficient × remaining depth², capped at 50,000 points.', 'Search', 20, 0, 200, 1, 'points'),
]);

export const DEFAULT_HEURISTICS = Object.freeze(Object.fromEntries(HEURISTIC_SETTINGS.map(entry => [entry.key, entry.default])));
const byKey = new Map(HEURISTIC_SETTINGS.map(entry => [entry.key, entry]));
const normalized = new WeakSet([DEFAULT_HEURISTICS]);

/** Validate once at API/search boundaries; normalized frozen objects are reusable. */
export function normalizeHeuristics(value) {
  if (value === undefined) return DEFAULT_HEURISTICS;
  if (value && normalized.has(value)) return value;
  if (!value || typeof value !== 'object' || (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)) {
    throw new TypeError('Heuristics must be a plain object of numeric settings.');
  }
  const result = { ...DEFAULT_HEURISTICS };
  for (const key of Reflect.ownKeys(value)) {
    const entry = byKey.get(key);
    if (!entry) throw new TypeError(`Unknown heuristic setting: ${String(key)}.`);
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    const number = descriptor.value;
    if (typeof number !== 'number' || !Number.isFinite(number)) throw new TypeError(`${key} must be a finite number.`);
    if (number < entry.min || number > entry.max) throw new RangeError(`${key} must be between ${entry.min} and ${entry.max}.`);
    if (key === 'quiescenceDepth' && !Number.isInteger(number)) throw new RangeError('quiescenceDepth must be a whole number of turns.');
    result[key] = number;
  }
  if (HEURISTIC_SETTINGS.every(entry => result[entry.key] === entry.default)) return DEFAULT_HEURISTICS;
  Object.freeze(result);
  normalized.add(result);
  return result;
}
