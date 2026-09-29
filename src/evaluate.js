import { raw } from './rules.js';
import { HEURISTIC_SETTINGS, normalizeHeuristics } from './heuristics.js';

// Centipawn values are deliberately conservative, hand-tuned starting values.
// A queen's extra dimensional freedom is worth substantially more than in 2D.
export const PIECE_VALUES = Object.freeze([0, 100, 355, 340, 550, 1150, 0, 900, 140, 370, 0, 450, 350]);
const VALUE_KEYS = [null, 'pawnValue', 'bishopValue', 'knightValue', 'rookValue', 'queenValue', null,
  'princessValue', 'brawnValue', 'commonKingValue', null, 'unicornValue', 'dragonValue'];
const pieceValueTables = new WeakMap();
const FEATURE_COMPONENTS = Object.freeze({
  ...Object.fromEntries(VALUE_KEYS.filter(Boolean).map(key => [key, 'material'])),
  ...Object.fromEntries(['pawnAdvanceWeight', 'pawnCenterWeight', 'doubledPawnWeight', 'isolatedPawnWeight',
    'passedPawnWeight', 'knightCenterWeight', 'bishopCenterWeight', 'pieceCenterWeight', 'royalCenterWeight',
    'mobilityWeight', 'developmentWeight', 'rookFileWeight'].map(key => [key, 'activity'])),
  ...Object.fromEntries(['shelterWeight', 'nearbyEnemyWeight', 'flankSafetyWeight', 'corridorWeight',
    'worstKingWeight'].map(key => [key, 'kingSafety'])),
  temporalPressureWeight: 'temporal', reserveWeight: 'timelines', overextensionWeight: 'timelines',
  weakBoardWeight: 'timelines', travelOpportunityWeight: 'travel',
});
const FEATURE_SETTINGS = HEURISTIC_SETTINGS.filter(entry => FEATURE_COMPONENTS[entry.key]);
const ROYAL_TYPES = new Set([6, 10]);
const KNIGHT_STEPS = [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]];
const AXES = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const DIAGONALS = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
const DIRECTIONS = [...AXES, ...DIAGONALS];
// Keep variant-piece geometry in sync with the rules library. These profiles
// describe attacks only: king safety and complete-turn legality remain search's
// responsibility, and pawns/brawns have separate directional capture rules.
// Small integer lookup tables avoid allocating and joining vectors for every
// attacker/royal pair. Leapers fit [-2, 2]; ray directions fit [-1, 1].
const stepKey = (l, t, r, f) => ((l + 2) * 5 + t + 2) * 25 + (r + 2) * 5 + f + 2;
const rayKey = (l, t, r, f) => ((l + 1) * 3 + t + 1) * 9 + (r + 1) * 3 + f + 1;
const MOVEMENT = Array.from({ length: PIECE_VALUES.length }, (_, type) => {
  const steps = new Uint8Array(625), rays = new Uint8Array(81);
  for (const v of raw.pieceFuncs.movePos(type * 2)) steps[stepKey(...v)] = 1;
  for (const v of raw.pieceFuncs.moveVecs(type * 2)) rays[rayKey(...v)] = 1;
  return { steps, rays };
});

/** Compile once per immutable profile; search reuses the numeric lookup table. */
export function pieceValuesFor(heuristics) {
  const settings = normalizeHeuristics(heuristics);
  let values = pieceValueTables.get(settings);
  if (!values) {
    values = Object.freeze(VALUE_KEYS.map(key => key ? settings[key] : 0));
    pieceValueTables.set(settings, values);
  }
  return values;
}

export function pieceValue(piece, heuristics) {
  return pieceValuesFor(heuristics)[Math.ceil(Math.abs(piece || 0) / 2)] || 0;
}

function owner(piece) { return Math.abs(piece) % 2; }
function signFor(color) { return color === 0 ? 1 : -1; }
function timelineCoordinate(index, even) {
  const value = index % 2 ? -(index + 1) / 2 : index / 2;
  return even && value > 0 ? value - 1 : value;
}

function timelineResources(board, heuristics) {
  // The rules allocate beyond each side's outermost index, even when a custom
  // position has holes. Index zero is shared; even/odd branches belong to W/B.
  const spent = [0, 0];
  for (let l = 1; l < board.length; l++) if (board[l]) spent[l % 2] = Math.ceil(l / 2);
  const available = spent.map((used, color) => Math.max(0, 1 + spent[1 - color] - used));
  const excess = spent.map((used, color) => Math.max(0, used - spent[1 - color] - 1));
  // The first unreciprocated branch spends our reserve and gives the opponent
  // another: a 180cp hurdle. Further inactive branches carry an extra burden.
  const reserve = 90 * (Math.min(4, available[0]) - Math.min(4, available[1])) * heuristics.reserveWeight;
  const overextension = 45 * (Math.min(3, excess[1]) - Math.min(3, excess[0])) * heuristics.overextensionWeight;
  return { available, reserve, overextension, score: reserve + overextension };
}

function temporalAttack(board, attacker, king, even) {
  // Half-turn boards of different colors cannot be connected by a move.
  if ((attacker.t - king.t) % 2) return false;
  const dl = king.line - attacker.line, dt = (king.t - attacker.t) / 2;
  const dr = king.r - attacker.r, df = king.f - attacker.f;
  if (attacker.type === 1 || attacker.type === 8) {
    const forward = attacker.color === 0 ? 1 : -1;
    if (dl === -forward && Math.abs(dt) === 1 && dr === 0 && df === 0) return true;
    // Brawns additionally capture across timeline/file, timeline/rank, and
    // past-time/rank planes. Match the pinned rules' directional captures.
    return attacker.type === 8 && (
      (dl === -forward && dt === 0 && ((dr === 0 && Math.abs(df) === 1) || (dr === forward && df === 0))) ||
      (dl === 0 && dt === -1 && dr === forward && df === 0)
    );
  }
  const movement = MOVEMENT[attacker.type];
  const al = Math.abs(dl), at = Math.abs(dt), ar = Math.abs(dr), af = Math.abs(df);
  const distance = Math.max(al, at, ar, af);
  if (distance <= 2 && movement.steps[stepKey(dl, dt, dr, df)]) return true;
  // A sliding vector must have equal nonzero components. Validate before
  // encoding signs, otherwise an off-ray target could alias a legal direction.
  if (!distance || (al && al !== distance) || (at && at !== distance) ||
      (ar && ar !== distance) || (af && af !== distance)) return false;
  const sl = Math.sign(dl), st = Math.sign(dt), sr = Math.sign(dr), sf = Math.sign(df);
  if (!movement.rays[rayKey(sl, st, sr, sf)]) return false;
  for (let offset = 1; offset < distance; offset++) {
    const line = raw.pieceFuncs.timelineMove(attacker.l, sl * offset, even);
    const square = board[line]?.[attacker.t + st * offset * 2]?.[attacker.r + sr * offset]?.[attacker.f + sf * offset];
    // Rays stop at occupied squares and at gaps in the multiverse. Leapers,
    // checked above, do not require intermediate boards or squares to exist.
    if (square !== 0) return false;
  }
  return true;
}

// Count ordinary nonroyal defenders of a king-zone pawn. A pawn defended only
// by its king (f2/f7 in the standard setup) is a particularly dangerous entry
// point. Defenders discount danger, but cannot make a temporal capture safe.
function pawnDefenders(squares, r, f, color) {
  let defenders = 0;
  for (const [dr, df] of DIRECTIONS) {
    for (let distance = 1; ; distance++) {
      const piece = squares[r + dr * distance]?.[f + df * distance];
      if (piece === undefined) break;
      if (!piece) continue;
      const type = Math.ceil(Math.abs(piece) / 2);
      if (owner(piece) === color && !ROYAL_TYPES.has(type)) {
        if (type === 1 || type === 8) {
          if (distance === 1 && dr === (color === 0 ? -1 : 1) && df !== 0) defenders++;
        } else if (MOVEMENT[type].rays[rayKey(0, 0, -dr, -df)] ||
          (distance <= 2 && MOVEMENT[type].steps[stepKey(0, 0, -dr * distance, -df * distance)])) defenders++;
      }
      break;
    }
  }
  for (const [dr, df] of KNIGHT_STEPS) {
    const piece = squares[r + dr]?.[f + df];
    if (piece && owner(piece) === color && Math.ceil(Math.abs(piece) / 2) === 3) defenders++;
  }
  return defenders;
}

function kingZone(squares, kings) {
  const targets = new Map();
  for (const king of kings) {
    targets.set(`${king.r},${king.f}`, { ...king, importance: 1.5 });
    for (const [dr, df] of DIRECTIONS) {
      const r = king.r + dr, f = king.f + df, piece = squares[r]?.[f];
      if (!piece || owner(piece) !== king.color || ![1, 8].includes(Math.ceil(Math.abs(piece) / 2))) continue;
      const defenders = pawnDefenders(squares, r, f, king.color);
      targets.set(`${r},${f}`, { ...king, r, f, pawn: true, defenders, importance: 1 / (1 + 0.5 * defenders) });
    }
  }
  return [...targets.values()];
}

function corridorRisk(timeline, latest, target, enemyTypes) {
  const forward = target.color === 0 ? 1 : -1;
  let risk = 0;
  for (const df of [-1, 0, 1]) {
    // A potential attacker comes back along time/rank or time/rank/file.
    // Only score corridors that an opposing slider can actually use.
    if (!enemyTypes.some(type => MOVEMENT[type].rays[rayKey(0, -1, -forward, -df)])) continue;
    let open = 0, enemyEntry = false;
    for (let distance = 1; distance <= 6; distance++) {
      const t = target.t + distance * 2;
      // Existing history is immutable. Project the frontier arrangement only
      // beyond recorded time; this rewards prophylaxis before an attack exists.
      const squares = t <= latest ? timeline[t] : timeline[latest];
      const piece = squares?.[target.r + forward * distance]?.[target.f + df * distance];
      if (piece === undefined) break; // Includes gaps in recorded history.
      if (piece && owner(piece) === target.color) break;
      open++;
      if (piece) {
        enemyEntry = Boolean(MOVEMENT[Math.ceil(Math.abs(piece) / 2)].rays[rayKey(0, -1, -forward, -df)]);
        break;
      }
    }
    // Short routes ending at the board edge offer fewer launch squares. A seal
    // within two steps closes the long approaches from the opponent's camp.
    risk += 24 * (enemyEntry ? 1 : Math.min(1, Math.max(0, open - 1) / 3)) * target.importance;
  }
  return risk;
}

function spatialActivity(board, r, f, type, color) {
  const steps = type === 3 ? KNIGHT_STEPS : type === 2 ? DIAGONALS : type === 4 ? AXES
    : type === 5 || type === 7 || ROYAL_TYPES.has(type) || type === 9 ? [...AXES, ...DIAGONALS] : [];
  const slider = [2, 4, 5, 7, 10].includes(type);
  let count = 0;
  for (const [dr, df] of steps) {
    let y = r + dr, x = f + df;
    while (board[y]?.[x] !== undefined) {
      const target = board[y][x];
      if (!target || owner(target) !== color) count++;
      if (target || !slider) break;
      y += dr; x += df;
    }
  }
  return count;
}

// Evaluate only the frontier of each timeline, never add up historical copies.
// Inactive timelines retain some value because they can reactivate later.
function evaluatePosition(position, heuristics, inspect = false, scoreOnly = false) {
  const settings = normalizeHeuristics(heuristics);
  const pieceValues = pieceValuesFor(settings);
  const { board } = position;
  const active = new Set(raw.boardFuncs.active(board));
  const even = raw.boardFuncs.isEvenTimeline(board);
  const resources = timelineResources(board, settings);
  const totals = { material: 0, activity: 0, kingSafety: 0, temporal: 0, timelines: 0, travel: 0 };
  const royals = [], entryPawns = [], attackers = [], frontier = [];
  let totalWeight = 0;
  const worstKing = [0, 0];
  const featureValues = inspect ? Object.fromEntries(FEATURE_SETTINGS.map(entry => [entry.key, 0])) : null;
  const boards = inspect ? [] : null;
  for (let l = 0; l < board.length; l++) {
    const timeline = board[l];
    if (!timeline?.length) continue;
    const t = timeline.length - 1, squares = timeline[t];
    if (!squares) continue;
    const weight = active.has(l) ? 1 : settings.inactiveWeight;
    const line = timelineCoordinate(l, even);
    totalWeight += weight;
    let material = 0, activity = 0, kingSafety = 0;
    const record = inspect ? (key, value) => { featureValues[key] += weight * value; } : null;
    const pawns = [[], []], pieces = [], kings = [];
    let phase = 0;
    for (let r = 0; r < squares.length; r++) {
      for (let f = 0; f < squares[r].length; f++) {
        const piece = squares[r][f];
        if (!piece) continue;
        const type = Math.ceil(Math.abs(piece) / 2), color = owner(piece), sign = signFor(color);
        const entry = { l, line, t, r, f, piece, type, color, weight };
        pieces.push(entry);
        const value = pieceValues[type] || 0;
        material += sign * value;
        if (record && VALUE_KEYS[type]) record(VALUE_KEYS[type], sign * value);
        if ([2, 3, 4, 5, 7].includes(type)) phase += value;
        if (type === 1 || type === 8) pawns[color].push(entry);
        if (ROYAL_TYPES.has(type)) { kings.push(entry); royals.push(entry); }
        attackers.push(entry);
      }
    }
    const middleGame = Math.min(1, phase / settings.phaseDivisor);
    const enemyTypes = [0, 1].map(color => [...new Set(pieces.filter(p => p.color !== color).map(p => p.type))]);
    const zoneRisk = [0, 0];
    // Keep the first king zone of each half-turn color as well as recent history:
    // a late blocker cannot erase an open route through early f-pawn snapshots.
    const sampleTimes = new Set([t]);
    for (const parity of [0, 1]) for (let first = parity; first <= t; first += 2) {
      if (timeline[first]) { sampleTimes.add(first); break; }
    }
    for (let past = t - 2, sampled = 0; past >= 0 && sampled < 6; past -= 2, sampled++) sampleTimes.add(past);
    // Travel setup can matter on either half-turn color. Inspect intervening
    // snapshots for concrete entry targets without changing the shelter sample.
    const entryTimes = new Set(sampleTimes);
    for (let past = t - 1, sampled = 0; past >= 0 && sampled < 12; past--, sampled++) entryTimes.add(past);
    for (const past of entryTimes) {
      const snapshot = timeline[past];
      if (!snapshot) continue;
      const pastKings = past === t ? kings : [];
      if (past !== t) for (let r = 0; r < snapshot.length; r++) for (let f = 0; f < snapshot[r].length; f++) {
        const piece = snapshot[r][f];
        if (ROYAL_TYPES.has(Math.ceil(Math.abs(piece) / 2))) pastKings.push({ l, line, t: past, r, f, color: owner(piece), weight: weight * settings.historicalPressureWeight });
      }
      const risk = [0, 0];
      for (const target of kingZone(snapshot, pastKings)) {
        if (sampleTimes.has(past)) risk[target.color] += corridorRisk(timeline, t, target, enemyTypes[target.color]);
        if (past < t && target.pawn && target.defenders === 0 && resources.available[1 - target.color] > 0) {
          // Being historical is what makes this an entry opportunity; unlike
          // royal pressure, it should not itself discount the target's value.
          entryPawns.push({ ...target, weight });
        }
      }
      // Shelter matters most with armies still on the board. Use the worst
      // snapshot rather than multiplying a weakness by its historical copies.
      for (const color of [0, 1]) zoneRisk[color] = Math.max(zoneRisk[color], risk[color] * middleGame * settings.corridorWeight);
    }
    for (const entry of pieces) {
      const { r, f, type, color, piece } = entry, sign = signFor(color);
      const rank = color === 0 ? r : squares.length - 1 - r;
      const center = (squares.length - 1) / 2;
      const centrality = Math.max(0, 4 - (Math.abs(r - center) + Math.abs(f - (squares[r].length - 1) / 2)) / 2);
      if (type === 1 || type === 8) {
        const advance = (rank * 7 + Math.max(0, rank - 3) ** 2 * 7) * settings.pawnAdvanceWeight;
        const centerScore = centrality * 3 * settings.pawnCenterWeight;
        activity += sign * (advance + centerScore);
        if (record) { record('pawnAdvanceWeight', sign * advance); record('pawnCenterWeight', sign * centerScore); }
        const sameFile = pawns[color].filter(p => p.f === f).length;
        if (sameFile > 1) {
          activity -= sign * 9 * settings.doubledPawnWeight;
          if (record) record('doubledPawnWeight', -sign * 9 * settings.doubledPawnWeight);
        }
        if (!pawns[color].some(p => Math.abs(p.f - f) === 1)) {
          activity -= sign * 9 * settings.isolatedPawnWeight;
          if (record) record('isolatedPawnWeight', -sign * 9 * settings.isolatedPawnWeight);
        }
        if (!pawns[1 - color].some(p => Math.abs(p.f - f) <= 1 && (color === 0 ? p.r > r : p.r < r))) {
          const passed = sign * (8 + rank * rank * 2) * settings.passedPawnWeight;
          activity += passed;
          if (record) record('passedPawnWeight', passed);
        }
      } else if (!ROYAL_TYPES.has(type)) {
        const mobility = spatialActivity(squares, r, f, type, color);
        const centralWeight = type === 3 ? 11 : type === 2 ? 6 : 3;
        const centralKey = type === 3 ? 'knightCenterWeight' : type === 2 ? 'bishopCenterWeight' : 'pieceCenterWeight';
        const mobilityScore = mobility * (type === 5 ? 2 : 4) * settings.mobilityWeight;
        const centerScore = centrality * centralWeight * settings[centralKey];
        activity += sign * (mobilityScore + centerScore);
        if (record) { record('mobilityWeight', sign * mobilityScore); record(centralKey, sign * centerScore); }
        if (piece < 0 && [2, 3].includes(type)) {
          const development = sign * 12 * middleGame * settings.developmentWeight;
          activity -= development;
          if (record) record('developmentWeight', -development);
        }
        if (type === 4 && !pawns[color].some(p => p.f === f)) {
          activity += sign * 14 * settings.rookFileWeight;
          if (record) record('rookFileWeight', sign * 14 * settings.rookFileWeight);
        }
      } else {
        let shield = 0, nearbyEnemies = 0;
        const forward = color === 0 ? 1 : -1;
        for (const df of [-1, 0, 1]) {
          const p = squares[r + forward]?.[f + df];
          if (p && owner(p) === color && [1, 8].includes(Math.ceil(Math.abs(p) / 2))) shield++;
        }
        for (const enemy of pieces) {
          if (enemy.color !== color && enemy.type !== 1 && Math.max(Math.abs(enemy.r - r), Math.abs(enemy.f - f)) <= 3) nearbyEnemies++;
        }
        const shelter = (3 - shield) * 12 * middleGame * settings.shelterWeight;
        const nearby = nearbyEnemies * 8 * settings.nearbyEnemyWeight;
        const flank = (rank === 0 && (f <= 2 || f >= squares[r].length - 2) ? 15 : 0) * settings.flankSafetyWeight;
        const risk = Math.max(0, shelter + nearby - flank);
        kingSafety -= sign * risk;
        const royalCenter = sign * centrality * 10 * (1 - middleGame) * settings.royalCenterWeight;
        activity += royalCenter;
        if (record) {
          record('shelterWeight', -sign * shelter);
          record('nearbyEnemyWeight', -sign * nearby);
          // Credit only the actual reduction: danger is floored at zero.
          record('flankSafetyWeight', sign * Math.min(flank, shelter + nearby));
          record('royalCenterWeight', royalCenter);
        }
        if (active.has(l)) worstKing[color] = Math.max(worstKing[color], risk + zoneRisk[color]);
      }
    }
    kingSafety += zoneRisk[1] - zoneRisk[0];
    if (record) record('corridorWeight', zoneRisk[1] - zoneRisk[0]);
    totals.material += weight * material;
    totals.activity += weight * activity;
    totals.kingSafety += weight * kingSafety;
    frontier.push({ l, t, weight, material });
    if (inspect) boards.push({ timeline: l, coordinate: line, turn: t, active: active.has(l), weight, material: weight * material,
      activity: weight * activity, kingSafety: weight * kingSafety, phase: middleGame });
    // Sample historical royal squares for potential time attacks. Historical
    // copies contribute pressure only, never additional material.
    for (let past = t - 2, sampled = 0; past >= 0 && sampled < 6; past -= 2, sampled++) {
      const snapshot = timeline[past];
      if (!snapshot) continue;
      for (let r = 0; r < snapshot.length; r++) for (let f = 0; f < snapshot[r].length; f++) {
        const piece = snapshot[r][f], type = Math.ceil(Math.abs(piece || 0) / 2);
        if (ROYAL_TYPES.has(type)) royals.push({ l, line, t: past, r, f, color: owner(piece), weight: weight * settings.historicalPressureWeight });
      }
    }
  }
  if (!totalWeight) return finishEvaluation(totals, settings, featureValues, boards, scoreOnly);
  for (const key of ['material', 'activity', 'kingSafety']) totals[key] /= totalWeight;
  if (inspect) {
    for (const key of Object.keys(featureValues)) featureValues[key] /= totalWeight;
    for (const entry of boards) for (const key of ['material', 'activity', 'kingSafety']) entry[key] /= totalWeight;
  }
  // One weak king can lose an otherwise healthy multiverse; averaging alone
  // would hide that weakness as more timelines are created.
  const worstKingScore = (worstKing[1] - worstKing[0]) * 0.45 * settings.worstKingWeight;
  totals.kingSafety += worstKingScore;
  if (inspect) featureValues.worstKingWeight = worstKingScore;

  // Reward unobstructed temporal attacks, including those by royal and fairy
  // pieces. This remains potential pressure, not a complete-turn check test.
  const pressure = [0, 0], travel = [0, 0];
  for (const attacker of attackers) {
    let best = 0;
    for (const king of royals) {
      if (king.color === attacker.color || (king.l === attacker.l && king.t === attacker.t)) continue;
      if (temporalAttack(board, attacker, king, even)) best = Math.max(best, 20 * Math.min(attacker.weight, king.weight) * settings.temporalPressureWeight);
    }
    pressure[attacker.color] += best;
    if (!resources.available[attacker.color] || ROYAL_TYPES.has(attacker.type)) continue;
    // On the other player's half-turn, the current arrangement can prepare a
    // route on our next board. Discount that projection: the reply may stop it.
    const ready = attacker.t % 2 === attacker.color;
    const source = ready ? attacker : { ...attacker, t: attacker.t + 1 };
    for (const target of entryPawns) {
      if (target.color === attacker.color || !temporalAttack(board, source, target, even)) continue;
      // Reserve is a scarce option. Count the best entry once, rather than
      // multiplying it by attackers, parallel boards, or historical copies.
      const value = 140 * (ready ? 1 : 0.5) * Math.min(1, pieceValues[attacker.type] / 340)
        * Math.min(attacker.weight, target.weight) * settings.travelOpportunityWeight;
      travel[attacker.color] = Math.max(travel[attacker.color], value);
    }
  }
  totals.temporal = Math.max(-100, Math.min(100, pressure[0] - pressure[1]));
  totals.travel = travel[0] - travel[1];
  totals.timelines = resources.score;
  if (inspect) {
    featureValues.temporalPressureWeight = totals.temporal;
    featureValues.travelOpportunityWeight = totals.travel;
    featureValues.reserveWeight = resources.reserve;
    featureValues.overextensionWeight = resources.overextension;
  }
  // Additional boards require defending additional kings. Penalize a frontier
  // material weakness that averaging would otherwise conceal.
  const activeFrontier = frontier.filter(b => active.has(b.l));
  if (activeFrontier.length > 1) {
    const low = Math.min(...activeFrontier.map(b => b.material));
    const high = Math.max(...activeFrontier.map(b => b.material));
    const weakBoard = (Math.min(0, low) + Math.max(0, high)) * 0.12 * settings.weakBoardWeight;
    totals.timelines += weakBoard;
    if (inspect) featureValues.weakBoardWeight = weakBoard;
  }
  return finishEvaluation(totals, settings, featureValues, boards, scoreOnly);
}

function finishEvaluation(totals, settings, featureValues, boards, scoreOnly) {
  totals.material *= settings.materialWeight;
  totals.activity *= settings.activityWeight;
  totals.kingSafety *= settings.kingSafetyWeight;
  totals.temporal *= settings.temporalWeight;
  totals.timelines *= settings.timelinesWeight;
  totals.travel *= settings.travelWeight;
  const total = Math.round(0 + totals.material + totals.activity + totals.kingSafety + totals.temporal + totals.timelines + totals.travel);
  // Recursive search consumes only the number. Allocate rounded component
  // reports only when the caller actually requested diagnostic information.
  if (scoreOnly) return total;
  const result = { ...Object.fromEntries(Object.entries(totals).map(([k, v]) => [k, Math.round(v)])), total };
  if (featureValues) {
    result.features = FEATURE_SETTINGS.map(({ key, label, description }) => ({ key, label, description,
      component: FEATURE_COMPONENTS[key], value: featureValues[key] * settings[`${FEATURE_COMPONENTS[key]}Weight`] }));
    result.boards = boards.map(entry => ({ ...entry, ...Object.fromEntries(['material', 'activity', 'kingSafety']
      .map(key => [key, entry[key] * settings[`${key}Weight`]])) }));
  }
  return result;
}

/** Rounded component scores; their individual rounding can differ from total. */
export function evaluateDetailed(position, heuristics) { return evaluatePosition(position, heuristics); }

/** Actual signed feature contributions after board and component weighting.
 * Features retain precision and sum to total before its final cp rounding.
 * Board contributions cover the averaged components; weakest-royal and
 * multiverse terms are global and are listed separately in features.
 */
export function inspectEvaluation(position, heuristics) { return evaluatePosition(position, heuristics, true); }

/** Positive values favor White. Mate scores are assigned by search only. */
export function evaluate(position, heuristics) { return evaluatePosition(position, heuristics, false, true); }
