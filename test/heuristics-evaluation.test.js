import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_HEURISTICS, HEURISTIC_SETTINGS, normalizeHeuristics } from '../src/heuristics.js';
import { evaluate, evaluateDetailed, inspectEvaluation, pieceValue, pieceValuesFor, PIECE_VALUES } from '../src/evaluate.js';
import { createPosition } from '../src/rules.js';

const empty = () => Array.from({ length: 8 }, () => Array(8).fill(0));
const kings = () => {
  const squares = empty();
  squares[0][4] = 12;
  squares[7][4] = 11;
  return squares;
};
const featuresOf = result => Object.fromEntries(result.features.map(feature => [feature.key, feature.value]));

// Recorded before introducing configurable evaluation. These are deliberately
// literal scores, not expectations derived by repeating the new implementation.
const baselineFixtures = [
  ['', { material: 0, activity: 0, kingSafety: 0, temporal: 0, timelines: 0, travel: 0, total: 0 }],
  ['1. e4', { material: 0, activity: 49, kingSafety: -26, temporal: 0, timelines: 0, travel: 0, total: 23 }],
  ['1. Nf3 / Nf6 2. d4 / d5 3. c3', { material: 0, activity: 11, kingSafety: 52, temporal: 0, timelines: 0, travel: 0, total: 63 }],
  ['1. e4 / d5 2. exd5', { material: 100, activity: 30, kingSafety: -23, temporal: 0, timelines: 0, travel: 0, total: 106 }],
];

test('default settings preserve pre-configuration evaluation scores', () => {
  for (const [pgn, expected] of baselineFixtures) {
    const position = createPosition({ pgn });
    assert.deepEqual(evaluateDetailed(position), expected, pgn || 'initial position');
    assert.deepEqual(evaluateDetailed(position, {}), expected);
    assert.deepEqual(evaluateDetailed(position, DEFAULT_HEURISTICS), expected);
    assert.equal(evaluate(position), expected.total);
  }
  for (let type = 0; type < PIECE_VALUES.length; type++) assert.equal(pieceValue(type * 2), PIECE_VALUES[type]);
});

test('heuristic schema is immutable and normalization fills defaults without retaining mutable input', () => {
  assert(Object.isFrozen(DEFAULT_HEURISTICS));
  assert(Object.isFrozen(HEURISTIC_SETTINGS));
  assert.equal(new Set(HEURISTIC_SETTINGS.map(entry => entry.key)).size, HEURISTIC_SETTINGS.length);
  for (const entry of HEURISTIC_SETTINGS) {
    assert(Object.isFrozen(entry));
    assert(entry.description && entry.label && entry.group);
    assert(entry.min <= entry.default && entry.default <= entry.max);
    assert(entry.step > 0);
    assert.equal(DEFAULT_HEURISTICS[entry.key], entry.default);
  }
  assert.equal(normalizeHeuristics(), DEFAULT_HEURISTICS);
  assert.equal(normalizeHeuristics({}), DEFAULT_HEURISTICS);
  const input = { pawnValue: 180, activityWeight: 0.5 };
  const normalized = normalizeHeuristics(input);
  assert(Object.isFrozen(normalized));
  assert.equal(normalizeHeuristics(normalized), normalized, 'Search reuses normalized settings without copying.');
  assert.equal(normalized.queenValue, DEFAULT_HEURISTICS.queenValue);
  input.pawnValue = 1;
  assert.equal(normalized.pawnValue, 180);
});

test('compiled piece values stay isolated from mutable inputs and other search profiles', () => {
  const input = { queenValue: 1500, pawnValue: 0 };
  const settings = normalizeHeuristics(input);
  const values = pieceValuesFor(settings);
  assert(Object.isFrozen(values));
  assert.equal(pieceValuesFor(settings), values);
  assert.equal(values[5], 1500);
  assert.equal(values[1], 0);
  assert.equal(values[6], 0);
  assert.equal(values[10], 0);
  input.queenValue = 2000;
  assert.equal(pieceValuesFor(input)[5], 2000);
  assert.equal(values[5], 1500);
  assert.deepEqual(pieceValuesFor(), PIECE_VALUES);
});

test('heuristic validation rejects malformed, unknown, nonfinite, and out-of-range values', () => {
  for (const value of [null, false, 3, 'settings', [], new Date(), new Map()]) assert.throws(() => normalizeHeuristics(value), /plain object/);
  for (const value of [NaN, Infinity, -Infinity, '1', null, undefined, true]) {
    assert.throws(() => normalizeHeuristics({ materialWeight: value }), /finite number/);
  }
  assert.throws(() => normalizeHeuristics({ unknown: 1 }), /Unknown heuristic/);
  assert.throws(() => normalizeHeuristics({ ['__proto__']: 1 }), /Unknown heuristic/);
  assert.throws(() => normalizeHeuristics({ [Symbol('key')]: 1 }), /Unknown heuristic/);
  assert.throws(() => normalizeHeuristics({ phaseDivisor: 0 }), /between/);
  assert.throws(() => normalizeHeuristics({ inactiveWeight: 1.01 }), /between/);
  assert.throws(() => normalizeHeuristics({ quiescenceDepth: 2.5 }), /whole number/);
  assert.throws(() => normalizeHeuristics({ quiescenceDepth: 9 }), /between/);
  assert.equal(normalizeHeuristics({ quiescenceDepth: 8 }).quiescenceDepth, 8);
  const accessor = Object.defineProperty({}, 'pawnValue', { get() { throw new Error('must not execute'); } });
  assert.throws(() => normalizeHeuristics(accessor), /finite number/);
});

test('custom material values affect current frontiers without counting historical copies', () => {
  const timeline = [kings(), kings(), kings()];
  timeline[0][3][3] = 10;
  timeline[2][2][2] = 2;
  const position = { board: [timeline], action: 0 };
  const settings = { pawnValue: 240, queenValue: 1800, activityWeight: 0, kingSafetyWeight: 0, temporalWeight: 0, travelWeight: 0 };
  assert.equal(evaluateDetailed(position, settings).material, 240);
  assert.equal(evaluate(position, settings), 240);
  assert.equal(pieceValue(-2, settings), 240);
  assert.equal(pieceValue(9, settings), 1800);
  assert.equal(pieceValue(12, settings), 0, 'Royal material stays zero.');
  timeline[0][4][4] = 9;
  assert.equal(evaluateDetailed(position, settings).material, 240);
});

test('feature controls change their actual score contributions and can disable evaluation', () => {
  const squares = kings();
  squares[3][3] = 2;
  squares[1][1] = -6;
  squares[6][6] = 9;
  const position = { board: [[squares]], action: 0 };
  const original = inspectEvaluation(position);
  const withoutIsolation = inspectEvaluation(position, { isolatedPawnWeight: 0 });
  assert.equal(featuresOf(original).isolatedPawnWeight, -9);
  assert.equal(featuresOf(withoutIsolation).isolatedPawnWeight, 0);
  assert.equal(withoutIsolation.activity, original.activity + 9);
  const withoutActivity = inspectEvaluation(position, { activityWeight: 0 });
  assert.equal(withoutActivity.activity, 0);
  assert(withoutActivity.features.filter(feature => feature.component === 'activity').every(feature => feature.value === 0));
  const disabled = Object.fromEntries(HEURISTIC_SETTINGS.filter(entry => entry.key.endsWith('Weight') && entry.group !== 'Position weighting').map(entry => [entry.key, 0]));
  const result = inspectEvaluation(position, disabled);
  assert.equal(result.total, 0);
  assert.equal(evaluate(position, disabled), result.total);
  assert(result.features.every(feature => feature.value === 0));
});

test('detailed contributions reconcile for multiple frontiers, historical travel, and custom weights', () => {
  const timeline = Array.from({ length: 5 }, kings);
  timeline[4][4][5] = 10;
  timeline[0][6][5] = 1;
  const board = [timeline];
  board[1] = [kings()];
  board[2] = [kings()];
  board[6] = [kings()];
  board[6][0][2][2] = 1;
  const position = { board, action: 0 };
  const before = JSON.stringify(position);
  for (const settings of [undefined, { pawnValue: 145, activityWeight: 0.35, kingSafetyWeight: 1.4, inactiveWeight: 0.45, reserveWeight: 0.7 }]) {
    const result = inspectEvaluation(position, settings);
    const { features, boards, ...score } = result;
    assert.deepEqual(score, evaluateDetailed(position, settings));
    assert.equal(evaluate(position, settings), result.total);
    assert(Math.abs(features.reduce((sum, feature) => sum + feature.value, 0) - result.total) <= 0.500001);
    for (const component of ['material', 'activity', 'kingSafety', 'temporal', 'timelines', 'travel']) {
      const sum = features.filter(feature => feature.component === component).reduce((sum, feature) => sum + feature.value, 0);
      assert(Math.abs(sum - result[component]) <= 0.500001, component);
    }
    assert.equal(boards.length, 4);
    assert(boards.some(board => !board.active));
    assert.equal(boards[0].turn, 4);
    assert.equal(boards[0].timeline, 0);
    const material = boards.reduce((sum, board) => sum + board.material, 0);
    assert(Math.abs(material - result.material) <= 0.500001);
    assert(features.every(feature => feature.label && feature.description && Number.isFinite(feature.value)));
  }
  assert.equal(JSON.stringify(position), before);
});

test('custom heuristic breakdown remains color-symmetric and does not multiply historical material', () => {
  const squares = kings();
  squares[3][3] = 2;
  squares[2][6] = -6;
  squares[6][4] = 9;
  const position = { board: [[squares]], action: 0 };
  const reflected = { board: [[squares.toReversed().map(row => row.map(piece => !piece ? 0 :
    Math.sign(piece) * (Math.abs(piece) + (Math.abs(piece) % 2 ? 1 : -1))))]], action: 1 };
  const settings = { knightValue: 465, mobilityWeight: 0.6, shelterWeight: 1.7, royalCenterWeight: 0.25 };
  const white = featuresOf(inspectEvaluation(position, settings));
  const black = featuresOf(inspectEvaluation(reflected, settings));
  for (const key of Object.keys(white)) assert(Math.abs(white[key] + black[key]) < 1e-9, key);
  position.board[0].push(structuredClone(squares), structuredClone(squares));
  assert.equal(featuresOf(inspectEvaluation(position, settings)).knightValue, white.knightValue);
});
