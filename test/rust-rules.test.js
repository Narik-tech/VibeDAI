import test from 'node:test';
import assert from 'node:assert/strict';
import {
  raw, applyMove, canSubmit, createPosition, generateActions, inCheck, parseMove,
  presentTimelines, pseudoMoves, validateAction,
} from '../src/rules.js';
import { nativeRequest, nativeTestOptions } from './rust-native-helper.js';

const promotions = [10, 9, 8, 7, 6, 5, 4, 3];
const empty = (size = 4) => Array.from({ length: size }, () => Array(size).fill(0));
const small = () => { const board = empty(); board[0][0] = 12; board[3][3] = 11; return board; };
const checked = () => { const board = small(); board[3][0] = 7; return board; };
const position = board => ({ board, action: 0, promotions });
const canonical = value => JSON.parse(JSON.stringify(value));
const sortedMoves = moves => moves.map(move => JSON.stringify(move)).sort();
const sortedStates = candidates => candidates.map(({ position: { board, action, promotions } }) =>
  JSON.stringify([action, board, promotions])).sort();

function fixtures() {
  const first = small(); first[1][1] = 4;
  const standard = createPosition();
  return [
    ['standard', standard],
    ['recorded time travel', createPosition({ pgn: '1. Nf3 / Nf6 2. Nc3 / Nc6' })],
    ['en passant', createPosition({ pgn: '1. e4 / a6 2. e5 / d5' })],
    ['castling', createPosition({ pgn: '[Board "Custom"]\n[k7/8/8/8/8/8/8/4K*2R*:0:1:w]' })],
    ['promotion', createPosition({ pgn: '[Board "Custom"]\n[Promotions "Q,R,B,N"]\n[7k/1P6/8/8/8/8/8/K7:0:1:w]' })],
    ['two checked boards', position([[checked()], null, [checked()]])],
    ['optional board and temporal arrival', position([[first], null, [small(), small(), small()]])],
    ['inactive board', position([[small()], null, [small()], null, [small()]])],
    ['submitted-ready partial turn', applyMove(standard, parseMove(standard, 'e4'))],
    ['partial multiboard evasion', applyMove(position([[checked()], null, [checked()]]), [[0, 0, 0, 0], [0, 0, 0, 1]])],
  ];
}

test('native move generation and royal safety match every starting variant and historical fixtures', nativeTestOptions, () => {
  for (const [label, start] of [...fixtures(), ...raw.metadataFuncs.variantDict
    .filter(([, variant]) => variant !== 'custom').map(([label, variant]) => [label, createPosition({ variant })])]) {
    const actual = nativeRequest({ type: 'moves', position: start });
    assert.deepEqual(sortedMoves(actual.moves), sortedMoves(pseudoMoves(start)), `${label}: pseudo-legal moves`);
    assert.deepEqual(actual.present, presentTimelines(start), `${label}: required timelines`);
    assert.deepEqual(actual.active, raw.boardFuncs.active(start.board), `${label}: active timelines`);
    assert.equal(actual.inCheck, inCheck(start), `${label}: royal safety`);
    assert.equal(actual.canSubmit, canSubmit(start), `${label}: submission legality`);
  }
});

test('native partial move application preserves castling, en passant, promotions and branching history', nativeTestOptions, () => {
  for (const [label, start] of fixtures().slice(0, 8)) {
    const moves = pseudoMoves(start);
    const selected = moves.filter(move => move.length !== 2 || move[1].length > 4
      || move[0][0] !== move[1][0] || move[0][1] !== move[1][1]);
    selected.push(...moves.slice(0, 2));
    // Bound the corpus while retaining temporal and special-move successors.
    for (const move of selected.slice(0, 12)) {
      const actual = nativeRequest({ type: 'apply', position: start, action: [move], submit: false });
      assert.deepEqual(actual, canonical(applyMove(start, move)), `${label}: ${JSON.stringify(move)}`);
    }
  }
});

test('native complete-turn generation matches JS for full rules, search policy and tactical filtering', nativeTestOptions, () => {
  const starts = fixtures().filter(([name]) => !['recorded time travel', 'en passant'].includes(name));
  for (const [label, start] of starts) for (const skipOptionalSpatial of [false, true]) for (const tacticalOnly of [false, true]) {
    const options = { skipOptionalSpatial, tacticalOnly, maxNodes: 200_000 };
    const actual = nativeRequest({ type: 'actions', position: start, options });
    assert.equal(actual.complete, true, `${label}: bounded diagnostic generation must exhaust the tree`);
    const expected = canonical([...generateActions(start, options)]);
    assert.deepEqual(sortedStates(actual.actions), sortedStates(expected),
      `${label}: complete successor states (restricted ${skipOptionalSpatial}, tactical ${tacticalOnly})`);
    assert.equal(new Set(sortedStates(actual.actions)).size, actual.actions.length, `${label}: deduplicated states`);
    for (const candidate of actual.actions) {
      assert.deepEqual(canonical(validateAction(start, candidate.moves)), candidate.position,
        `${label}: every native recommendation validates through the JS application rules`);
    }
  }
});

test('native generation reports incomplete work and never confuses a budget with terminality', nativeTestOptions, () => {
  const result = nativeRequest({ type: 'actions', position: createPosition(), options: { maxNodes: 0 } });
  assert.equal(result.complete, false);
  assert.deepEqual(result.actions, []);
});
