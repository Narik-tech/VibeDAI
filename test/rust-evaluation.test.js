import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { DEFAULT_CLASSICAL_ENGINE_PATH } from '../src/rust-engine.js';
import { evaluateDetailed, inspectEvaluation } from '../src/evaluate.js';
import { HEURISTIC_SETTINGS } from '../src/heuristics.js';
import { createPosition } from '../src/rules.js';

const executable = process.env.CLASSICAL_ENGINE_PATH || DEFAULT_CLASSICAL_ENGINE_PATH;
const native = { skip: existsSync(executable) ? false : 'Build the native engine with npm run build:classical.' };
const empty = (height = 8, width = 8) => Array.from({ length: height }, () => Array(width).fill(0));
function kings() {
  const board = empty();
  board[0][4] = 12;
  board[7][4] = 11;
  return board;
}
function request(type, position, heuristics) {
  const child = spawnSync(executable, [], {
    input: JSON.stringify({ type, position, options: { heuristics } }) + '\n',
    encoding: 'utf8', windowsHide: true, timeout: 10_000, maxBuffer: 4_000_000,
  });
  assert.ifError(child.error);
  assert.equal(child.status, 0, child.stderr);
  const messages = child.stdout.trim().split('\n').map(line => JSON.parse(line));
  const result = messages.find(message => message.type === 'result');
  assert(result, JSON.stringify(messages));
  return result.result;
}
// JSON normalizes JavaScript's -0, just as the native wire protocol does.
const wire = value => JSON.parse(JSON.stringify(value));
function equalEvaluation(position, heuristics, label) {
  assert.deepEqual(request('evaluate', position, heuristics), wire(evaluateDetailed(position, heuristics)), label);
}

test('native classical evaluation preserves opening scores and custom profiles', native, () => {
  for (const pgn of ['', '1. e4', '1. Nf3 / Nf6 2. d4 / d5 3. c3', '1. e4 / d5 2. exd5']) {
    const position = createPosition({ pgn });
    for (const settings of [undefined, { pawnValue: 145, activityWeight: 0.35, kingSafetyWeight: 1.4, reserveWeight: 0.7 }]) {
      equalEvaluation(position, settings, pgn || 'initial position');
    }
  }
});

test('native evaluation preserves sparse timelines, historical blockers, and overlapping royal zones', native, () => {
  const history = Array.from({ length: 19 }, () => {
    const squares = empty();
    for (const [r, f, piece] of [[0, 3, 12], [0, 5, 20], [1, 4, 2], [7, 4, 11], [6, 3, 1], [5, 7, 9]]) squares[r][f] = piece;
    return squares;
  });
  history[2] = null;
  history[9] = null;
  history[18][4][5] = 10;
  equalEvaluation({ board: [history], action: 0 }, undefined, 'overlapping royal history');
  for (const indices of [[0], [0, 1], [0, 2, 4], [0, 6], [1, 2], [1, 2, 3], [1, 2, 4]]) {
    const board = [];
    for (const index of indices) board[index] = [kings()];
    equalEvaluation({ board, action: 0 }, { inactiveWeight: 0.45 }, `timelines ${indices}`);
  }
  for (const blocker of [0, 1, 2, null]) {
    const timeline = Array.from({ length: 5 }, kings);
    timeline[4][4][5] = 10;
    timeline[0][6][5] = 1;
    if (blocker === null) timeline[2] = null;
    else timeline[2][5][5] = blocker;
    equalEvaluation({ board: [timeline], action: 0 }, undefined, `historical blocker ${blocker}`);
  }
});

test('native evaluation matches every piece family on generated sparse multiverses', native, () => {
  // Deterministic generated positions exercise geometry and weighting without
  // expecting them to be reachable by legal play. They remain valid wire boards.
  let seed = 5821;
  const random = n => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; };
  for (let sample = 0; sample < 20; sample++) {
    const board = [];
    for (const line of sample % 2 ? [1, 2, 4] : [0, 1, 2, 6]) {
      const timeline = [];
      for (let t = 0; t < 5; t++) {
        if (t !== 4 && random(7) === 0) { timeline.push(null); continue; }
        const squares = kings();
        for (let i = 0; i < 12; i++) {
          const rank = 1 + random(6), file = random(8);
          squares[rank][file] = (1 + random(24)) * (random(3) ? 1 : -1);
        }
        timeline.push(squares);
      }
      board[line] = timeline;
    }
    const settings = sample % 2 ? {
      phaseDivisor: 8000, pawnValue: 135, knightValue: 465, mobilityWeight: 0.6,
      shelterWeight: 1.7, royalCenterWeight: 0.25, inactiveWeight: 0.45,
      temporalPressureWeight: 1.35, travelOpportunityWeight: 0.7,
    } : undefined;
    equalEvaluation({ board, action: sample % 2 }, settings, `generated sample ${sample}`);
  }
});

test('native diagnostic features retain precision and reconcile to component scores', native, () => {
  const timeline = Array.from({ length: 5 }, kings);
  timeline[4][4][5] = 10;
  timeline[0][6][5] = 1;
  const board = [timeline, [kings()], [kings()], null, null, null, [kings()]];
  board[6][0][2][2] = 1;
  const position = { board, action: 0 };
  const settings = { pawnValue: 145, activityWeight: 0.35, kingSafetyWeight: 1.4, inactiveWeight: 0.45, reserveWeight: 0.7 };
  const actual = request('inspect', position, settings);
  const expected = inspectEvaluation(position, settings);
  for (const [index, feature] of expected.features.entries()) {
    const nativeFeature = actual.features[index];
    assert.equal(nativeFeature.key, feature.key);
    assert.equal(nativeFeature.description, feature.description);
    assert.equal(nativeFeature.label, feature.label);
    assert.equal(nativeFeature.component, feature.component);
    assert(Math.abs(nativeFeature.value - feature.value) < 1e-9, feature.key);
  }
  assert.deepEqual(actual.boards.map(({ phase, ...board }) => board), wire(expected.boards.map(({ phase, ...board }) => board)));
  const disabled = Object.fromEntries(HEURISTIC_SETTINGS.filter(entry => entry.key.endsWith('Weight') && entry.group !== 'Position weighting').map(entry => [entry.key, 0]));
  equalEvaluation(position, disabled, 'all scoring disabled');
});
